import time
import tqdm
import torch
from model import ByteTransformer, ByteCorpus
from model import complete, next_byte_cross_entropy_loss


def train(device):
    print("chosen device:", device)
    
    XLA = (device == 'xla')
    if XLA:
        print("set environment variables expected by torch_xla...")
        import os
        os.environ['PJRT_DEVICE'] = "TPU"
        print("importing torch_xla...")
        import torch_xla.core.xla_model as xm
        print("initialising default xla device...")
        device = xm.xla_device()

    print("loading data...")
    data = ByteCorpus(
        path="data/sherlock-ascii.txt",
        device=device,
    )

    print("initialising model...")
    model = ByteTransformer(
        max_context_length=128,
        embed_size=32,
        mlp_size=64,
        num_heads=4,
        num_layers=4,
        device=device,
    )
    model.train()
    
    # initialising training stuff
    num_training_steps = 50000
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )
    
    if XLA: xm.mark_step()

    print("training model...")
    for steps in tqdm.trange(num_training_steps):
        if XLA: xm.mark_step()
        batch = data.get_training_batch(seq_length=128, batch_size=32)
        logits = model(batch)
        loss = next_byte_cross_entropy_loss(batch, logits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if XLA: xm.mark_step()

        # evaluate periodically
        if steps % 100 == 0:
            tqdm.tqdm.write(f"eval at step {steps:>6d}")
            if XLA: xm.mark_step()
            model.eval()
            # batch loss
            tqdm.tqdm.write(f"  training loss: {loss.item():>6.3f}")
            # test loss
            eval_batch = data.get_testing_batch(seq_length=128, batch_size=256)
            with torch.no_grad():
                logits = model(eval_batch)
            loss = next_byte_cross_entropy_loss(eval_batch, logits)
            tqdm.tqdm.write(f"  testing loss:  {loss.item():>6.3f}")
            # prompt
            with torch.no_grad():
                ctn = complete(model, '"Elementary, my dear', max_bytes=32)
            tqdm.tqdm.write(f"  continuation:  {ctn!r}")
            model.train()
            if XLA: xm.mark_step()

    print("done!")
    print("generating passage from model...")
    model.to('cpu')
    with torch.no_grad():
        ctn = complete(model, '"Elementary, my dear', max_bytes=256)
    if XLA: xm.mark_step()
    for c in ctn:
        print(c, end="", flush=True)
        time.sleep(0.06125)


if __name__ == "__main__":
    # command line arguments
    import sys
    if sys.argv[1:]:
        device = sys.argv[1]
    else:
        device = 'cpu'

    # main script
    train(device=device)
