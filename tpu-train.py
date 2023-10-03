import sys
import tqdm
import torch
from model import ByteTransformer, ByteCorpus
from model import complete, next_byte_cross_entropy_loss


if sys.argv[1:]:
    DEVICE = sys.argv[1]
else:
    DEVICE = 'cpu'

XLA = (DEVICE == 'xla')
if XLA:
    print("importing torch_xla...")
    import os
    os.environ['PJRT_DEVICE'] = "TPU"
    import torch_xla.core.xla_model as xm
    print("initialising default xla device...")
    DEVICE = xm.xla_device()


def train():
    print("loading data...")
    data = ByteCorpus(
        path="data/sherlock-ascii.txt",
        device=DEVICE,
    )

    print("initialising model...")
    model = ByteTransformer(
        max_context_length=128,
        embed_size=32,
        mlp_size=64,
        num_heads=4,
        num_layers=4,
        device=DEVICE,
    )
    model.train()
    
    # initialising training stuff
    num_training_steps = 100000
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=0.0003,
        anneal_strategy='linear',
        total_steps=num_training_steps,
        pct_start=0.50,
        div_factor=num_training_steps / 2,
        final_div_factor=num_training_steps / 2,
        cycle_momentum=False, # N/A for adam but required to avoid error
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
        scheduler.step()
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

    print("done")


if __name__ == "__main__":
    train()
