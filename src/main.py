import torch

from utils.common import checkpoint
import data
import model
import loss
from option import args
from trainer import Trainer


torch.manual_seed(args.seed)
checkpoint_id = checkpoint(args)

if checkpoint_id.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint_id)
    loss = loss.Loss(args, checkpoint_id) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint_id)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint_id.done()