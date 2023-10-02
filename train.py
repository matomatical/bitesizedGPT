import tqdm

import torch

from model import ByteTransformer, ByteCorpus, next_byte_cross_entropy_loss


DEVICE = 'mps'

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
        num_layers=16,
        device=DEVICE,
    )
    model.train()
    
    # initialising training stuff
    num_training_steps = 100000
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    print("training model...")
    for steps in tqdm.trange(num_training_steps):
        batch = data.get_training_batch(seq_length=128, batch_size=32)
        logits = model(batch)
        loss = next_byte_cross_entropy_loss(batch, logits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # evaluate periodically
        if steps % 100 == 0:
            tqdm.tqdm.write(f"eval at step {steps:>6d}")
            tqdm.tqdm.write(f"  training loss: {loss.item():>6.3f}")
            model.eval()
            eval_batch = data.get_testing_batch(seq_length=128, batch_size=256)
            logits = model(eval_batch)
            loss = next_byte_cross_entropy_loss(eval_batch, logits)
            tqdm.tqdm.write(f"  testing loss:  {loss.item():>6.3f}")
    print("done")


if __name__ == "__main__":
    train()
    
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer=optimizer,
    #     max_lr=0.001,
    #     anneal_strategy='linear',
    #     total_steps=num_training_steps,
    #     pct_start=0.50,
    #     div_factor=num_training_steps / 2,
    #     final_div_factor=num_training_steps / 2,
    #     cycle_momentum=False, # N/A for adam but required to avoid error
    # )
