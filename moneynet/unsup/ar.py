import os
import logging
import json

import numpy as np

import torch

from tqdm import tqdm

from moneynet.nets.stn_ar import Net
from moneynet.utils.pikachu_dataset import Pikachu


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
    model = model.to(device)

    # Setup an optimizer
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), args.lr, eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    # Training dataset
    if args.online_learning:
        model.train()
        # ToDo(j-pong): shuffle with range class
        epoch_lens = dataset.__len__()
        for batch_idx in tqdm(range(args.epochs * epoch_lens)):
            idx = int(batch_idx/epoch_lens)
            data, target = dataset.__getitem__(idx)

            data = np.expand_dims(data.T, axis=0)
            target = np.expand_dims(target.T, axis=0)

            data, target = torch.from_numpy(data).to(device), torch.from_numpy(target).to(device)

            optimizer.zero_grad()
            loss = model(data, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 500 == 0:
                print(float(loss))
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        # Model training loop
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            optimizer.step()
            if batch_idx % 500 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    args.epochs, batch_idx * len(data), len(train_loader.dataset),
                                 100. * batch_idx / len(train_loader), loss.item()))

# @torch.no_grad()
# def test(args):
#     # show arguments
#     for key in sorted(vars(args).keys()):
#         logging.info('args: ' + key + ': ' + str(vars(args)[key]))
#
#     # define model
#     model_class = dynamic_import(train_args.model_module)
#     model = model_class(idim, odim, train_args)
#     assert isinstance(model, TTSInterface)
#     logging.info(model)
#
#     # load trained model parameters
#     logging.info('reading model parameters from ' + args.model)
#     torch_load(args.model, model)
#     model.eval()
#
#     # set torch device
#     device = torch.device("cuda" if args.ngpu > 0 else "cpu")
#     model = model.to(device)
#
#     # read json data
#     with open(args.json, 'rb') as f:
#         js = json.load(f)['utts']
#
#     # check directory
#     outdir = os.path.dirname(args.out)
#     if len(outdir) != 0 and not os.path.exists(outdir):
#         os.makedirs(outdir)
#
#     load_inputs_and_targets = LoadInputsAndTargets(
#         mode='tts', load_input=False, sort_in_input_length=False,
#         use_speaker_embedding=train_args.use_speaker_embedding,
#         preprocess_conf=train_args.preprocess_conf
#         if args.preprocess_conf is None else args.preprocess_conf,
#         preprocess_args={'train': False}  # Switch the mode of preprocessing
#     )
#
#     # define function for plot prob and att_ws
#     def _plot_and_save(array, figname, figsize=(6, 4), dpi=150):
#         import matplotlib.pyplot as plt
#         shape = array.shape
#         if len(shape) == 1:
#             # for eos probability
#             plt.figure(figsize=figsize, dpi=dpi)
#             plt.plot(array)
#             plt.xlabel("Frame")
#             plt.ylabel("Probability")
#             plt.ylim([0, 1])
#         elif len(shape) == 2:
#             # for tacotron 2 attention weights, whose shape is (out_length, in_length)
#             plt.figure(figsize=figsize, dpi=dpi)
#             plt.imshow(array, aspect="auto")
#             plt.xlabel("Input")
#             plt.ylabel("Output")
#         elif len(shape) == 4:
#             # for transformer attention weights, whose shape is (#leyers, #heads, out_length, in_length)
#             plt.figure(figsize=(figsize[0] * shape[0], figsize[1] * shape[1]), dpi=dpi)
#             for idx1, xs in enumerate(array):
#                 for idx2, x in enumerate(xs, 1):
#                     plt.subplot(shape[0], shape[1], idx1 * shape[1] + idx2)
#                     plt.imshow(x, aspect="auto")
#                     plt.xlabel("Input")
#                     plt.ylabel("Output")
#         else:
#             raise NotImplementedError("Support only from 1D to 4D array.")
#         plt.tight_layout()
#         if not os.path.exists(os.path.dirname(figname)):
#             # NOTE: exist_ok = True is needed for parallel process decoding
#             os.makedirs(os.path.dirname(figname), exist_ok=True)
#         plt.savefig(figname)
#         plt.close()
#
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         data, target = data.to(device), target.to(device)
#         output = model(data)
#
#         # sum up batch loss
#         test_loss += F.nll_loss(output, target, size_average=False).item()
#         # get the index of the max log-probability
#         pred = output.max(1, keepdim=True)[1]
#         correct += pred.eq(target.view_as(pred)).sum().item()
#
#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
#           .format(test_loss, correct, len(test_loader.dataset),
#                   100. * correct / len(test_loader.dataset)))
