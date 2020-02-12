import os
import logging
import json

import torch

from tqdm import tqdm

from moneynet.nets.simnn import Net
from moneynet.utils.pikachu_dataset import Pikachu


def train_core(train_loader, optimizer, device, model):
    for samples in tqdm(train_loader):
        data = samples['input'].to(device)
        target = samples['target'].to(device)

        optimizer.zero_grad()
        loss = model(data, target)
        loss.backward()
        optimizer.step()
    return float(loss)


def train(args):
    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    dataset = Pikachu(root=args.indir)

    # reverse input and output dimension
    idim, odim = map(int, dataset.__dims__())
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # specify model architecture
    model = Net(idim, odim)
    logging.info(model)

    # check the use of multi-gpu
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        if args.batch_size != 0:
            logging.warning('batch size is automatically increased (%d -> %d)' % (
                args.batch_size, args.batch_size * args.ngpu))
            args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32
    model = model.to(device=device, dtype=dtype)
    model = model.to(device)

    # Setup an optimizer
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), args.lr, eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    # Training dataset
    model.train()
    for i in range(args.epochs):
        loss = train_core(train_loader, optimizer, device, model)
        print("{} epoch is end and loss is {}".format(i, loss))