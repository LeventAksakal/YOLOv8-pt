import argparse
import copy
import csv
import os
import warnings
import matplotlib.pyplot as plt
import cv2

import numpy
import torch
import tqdm
import yaml
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")


def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params["lrf"]) + params["lrf"]

    return fn


def train(args, params):
    # Model
    model = nn.yolo_v8_n(len(params["names"].values())).cuda()

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params["weight_decay"] *= args.batch_size * args.world_size * accumulate / 64

    p = [], [], []
    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, torch.nn.Parameter):
            p[2].append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            p[1].append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, torch.nn.Parameter):
            p[0].append(v.weight)

    optimizer = torch.optim.SGD(p[2], params["lr0"], params["momentum"], nesterov=True)

    optimizer.add_param_group({"params": p[0], "weight_decay": params["weight_decay"]})
    optimizer.add_param_group({"params": p[1]})
    del p

    # Scheduler
    lr = learning_rate(args, params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    # filenames = []
    # with open("../Dataset/COCO/train2017.txt") as reader:
    #     for filename in reader.readlines():
    #         filename = filename.rstrip().split("/")[-1]
    #         filenames.append("../Dataset/COCO/images/train2017/" + filename)
    directory = "dataset\\images\\train"
    filenames = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]

    dataset = Dataset(filenames, args.input_size, params, True)

    if args.world_size <= 1:
        sampler = None
    else:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(
        dataset,
        args.batch_size,
        sampler is None,
        sampler,
        num_workers=8,
        pin_memory=True,
        collate_fn=Dataset.collate_fn,
    )

    if args.world_size > 1:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # Start training
    best = 0
    num_batch = len(loader)
    amp_scale = torch.cuda.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)
    num_warmup = max(round(params["warmup_epochs"] * num_batch), 1000)
    with open("weights/step.csv", "w") as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=["epoch", "mAP@50", "mAP"])
            writer.writeheader()
        for epoch in range(args.epochs):
            model.train()

            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            m_loss = util.AverageMeter()
            if args.world_size > 1:
                sampler.set_epoch(epoch)
            p_bar = enumerate(loader)
            if args.local_rank == 0:
                print(("\n" + "%10s" * 3) % ("epoch", "memory", "loss"))
            if args.local_rank == 0:
                p_bar = tqdm.tqdm(p_bar, total=num_batch)  # progress bar

            optimizer.zero_grad()

            for i, (samples, targets, _) in p_bar:
                x = i + num_batch * epoch  # number of iterations
                samples = samples.cuda().float() / 255
                targets = targets.cuda()

                # Warmup
                if x <= num_warmup:
                    xp = [0, num_warmup]
                    fp = [1, 64 / (args.batch_size * args.world_size)]
                    accumulate = max(1, numpy.interp(x, xp, fp).round())
                    for j, y in enumerate(optimizer.param_groups):
                        if j == 0:
                            fp = [params["warmup_bias_lr"], y["initial_lr"] * lr(epoch)]
                        else:
                            fp = [0.0, y["initial_lr"] * lr(epoch)]
                        y["lr"] = numpy.interp(x, xp, fp)
                        if "momentum" in y:
                            fp = [params["warmup_momentum"], params["momentum"]]
                            y["momentum"] = numpy.interp(x, xp, fp)

                # Forward
                with torch.cuda.amp.autocast():
                    outputs = model(samples)  # forward
                loss = criterion(outputs, targets)

                m_loss.update(loss.item(), samples.size(0))

                loss *= args.batch_size  # loss scaled by batch_size
                loss *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(loss).backward()

                # Optimize
                if x % accumulate == 0:
                    amp_scale.unscale_(optimizer)  # unscale gradients
                    util.clip_gradients(model)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                # Log
                if args.local_rank == 0:
                    memory = f"{torch.cuda.memory_reserved() / 1E9:.3g}G"  # (GB)
                    s = ("%10s" * 2 + "%10.4g") % (
                        f"{epoch + 1}/{args.epochs}",
                        memory,
                        m_loss.avg,
                    )
                    p_bar.set_description(s)

                del loss
                del outputs

            # Scheduler
            scheduler.step()

            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema)
                writer.writerow(
                    {
                        "mAP": str(f"{last[1]:.3f}"),
                        "epoch": str(epoch + 1).zfill(3),
                        "mAP@50": str(f"{last[0]:.3f}"),
                    }
                )
                f.flush()

                # Update best mAP
                if last[1] > best:
                    best = last[1]

                # Save model
                ckpt = {"model": copy.deepcopy(ema.ema).half()}

                # Save last, best and delete
                torch.save(ckpt, "./weights/last.pt")
                if best == last[1]:
                    torch.save(ckpt, "./weights/best.pt")
                del ckpt

    if args.local_rank == 0:
        util.strip_optimizer("./weights/best.pt")  # strip optimizers
        util.strip_optimizer("./weights/last.pt")  # strip optimizers

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, params, model=None):
    # Load dataset
    directory = "dataset\\images\\valid"
    filenames = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    dataset = Dataset(filenames, args.input_size, params, False)
    # plot image
    # image = dataset[0][0].permute(1, 2, 0).cpu().numpy()
    # plt.imshow(image)
    # plt.show()

    loader = data.DataLoader(
        dataset, 8, False, num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn
    )
    if model is None:
        model = torch.load("./weights/best.pt", map_location="cuda")["model"].float()

    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0.0
    m_rec = 0.0
    map50 = 0.0
    mean_ap = 0.0
    metrics = []

    p_bar = tqdm.tqdm(loader, desc=("%10s" * 3) % ("precision", "recall", "mAP"))
    for i, (samples, targets, shapes) in enumerate(p_bar):
        samples = samples.cuda()
        targets = targets.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255  # 0 - 255 to 0.0 - 1.0
        _, _, height, width = samples.shape  # batch size, channels, height, width

        # Inference
        outputs = model(samples)
        # NMS
        targets[:, 2:] *= torch.tensor(
            (width, height, width, height)
        ).cuda()  # to pixels
        outputs = util.non_max_suppression(outputs, 0.001, 0.65)

        # Save predictions on the validation set as csv
        # save_predictions(outputs, shapes, (640, 640))

        # Save images with predictions on the validation set
        # save_images(samples, outputs)

        # Metrics
        for i, output in enumerate(outputs):
            labels = targets[targets[:, 0] == i, 1:]
            correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if labels.shape[0]:
                    metrics.append((correct, *torch.zeros((3, 0)).cuda()))
                continue

            detections = output.clone()
            util.scale(
                detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1]
            )

            # Evaluate
            if labels.shape[0]:
                tbox = labels[:, 1:5].clone()  # target boxes
                tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2  # top left x
                tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2  # top left y
                tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2  # bottom right x
                tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2  # bottom right y
                util.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])

                correct = numpy.zeros((detections.shape[0], iou_v.shape[0]))
                correct = correct.astype(bool)

                t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                iou = util.box_iou(t_tensor[:, 1:], detections[:, :4])
                correct_class = t_tensor[:, 0:1] == detections[:, 5]
                for j in range(len(iou_v)):
                    x = torch.where((iou >= iou_v[j]) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat(
                            (torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1
                        )
                        matches = matches.cpu().numpy()
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[
                                numpy.unique(matches[:, 1], return_index=True)[1]
                            ]
                            matches = matches[
                                numpy.unique(matches[:, 0], return_index=True)[1]
                            ]
                        correct[matches[:, 1].astype(int), j] = True
                correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
            metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)

    # Print results
    print("%10.3g" * 3 % (m_pre, m_rec, mean_ap))

    # Return results
    model.float()  # for training
    return map50, mean_ap


def reverse_transform(image, shapes):

    original_shape, transform_info = shapes
    scale, pad = transform_info

    image


def save_predictions(predictions, shapes_list, output_size, output_path="."):
    with open(
        os.path.join(output_path, "test_predictions.csv"), "w", newline=""
    ) as csvfile:
        fieldnames = ["image_index", "conf", "cls", "x", "y", "w", "h"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, (output, shapes) in enumerate(zip(predictions, shapes_list)):
            print(f"Processing prediction {i + 1}/{len(predictions)}")
            if output is not None:
                boxes = output[:, :4].cpu().numpy()
                reversed_boxes = reverse_transform(boxes, shapes, output_size)
                for j, box in enumerate(reversed_boxes):
                    x1, y1, x2, y2 = box
                    x = (x1 + x2) / 2
                    y = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    conf = output[j, 4].item()
                    cls = int(output[j, 5].item())
                    writer.writerow(
                        {
                            "image_index": f"{i}",
                            "conf": conf,
                            "cls": cls,
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h,
                        }
                    )


def save_images(images, predictions, output_path="."):
    for i, output in enumerate(predictions):
        image = images[i].cpu().numpy().transpose(1, 2, 0) * 255
        image = cv2.cvtColor(image.astype(numpy.uint8), cv2.COLOR_RGB2BGR)

        if output is not None and len(output):
            for *box, conf, cls in output:
                x1, y1, x2, y2 = map(int, box)
                label = f"{cls:.0f} {conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Save image
        output_directory = "output_images"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        output_path = os.path.join(output_directory, f"image_{i}.jpg")
        cv2.imwrite(output_path, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", default=640, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    args.local_rank = int(os.getenv("LOCAL_RANK", 0))
    args.world_size = int(os.getenv("WORLD_SIZE", 1))

    if args.world_size > 1:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if args.local_rank == 0:
        if not os.path.exists("weights"):
            os.makedirs("weights")

    util.setup_seed()
    util.setup_multi_processes()

    with open(os.path.join("utils", "args.yaml"), errors="ignore") as f:
        params = yaml.safe_load(f)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)


if __name__ == "__main__":
    main()
