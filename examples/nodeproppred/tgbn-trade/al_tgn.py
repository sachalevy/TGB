import timeit
import argparse
from tqdm import tqdm
import torch
import copy
from pathlib import Path
import json
import matplotlib.pyplot as plt
from concurrent import futures
import multiprocessing as mp


from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
import numpy as np
from tqdm import tqdm

from modules.decoder import NodePredictor
from modules.emb_module import GraphAttentionEmbedding
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from tgb.nodeproppred.evaluate import Evaluator
from tgb.utils.utils import set_random_seed
from examples.nodeproppred import utils as nprop_utils

parser = argparse.ArgumentParser(
    description="parsing command line arguments as hyperparameters"
)
parser.add_argument("-s", "--seed", type=int, default=1, help="random seed to use")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--count", type=int, default=2)
parser.add_argument("--min_samples", type=int, default=100)
parser.add_argument("--max_samples", type=int, default=10000)
parser.add_argument("--name", type=str, default="tgbn-trade")
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--repeats", type=int, default=1)

parser.parse_args()
args = parser.parse_args()
name = args.name

# setting random seed
seed = int(args.seed)  # 1,2,3,4,5
torch.manual_seed(seed)
set_random_seed(seed)

# hyperparameters
lr = 0.0001

fig_dirpath = Path("figs/")
fig_dirpath.mkdir(exist_ok=True, parents=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_curve(x, y, output_filepath, x_axis_label="Epochs", legends=None):
    plt.figure(figsize=(10, 6))  # Set the figure size

    if legends:
        for _y, legend in zip(y, legends):
            plt.plot(
                x,
                _y,
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=6,
                label=legend,
            )  # Add markers and adjust line

    else:
        plt.plot(
            x, y, color="#e34a33", marker="o", linestyle="-", linewidth=2, markersize=6
        )  # Add markers and adjust line
    plt.title("Performance Curve")  # Add a title
    plt.xlabel(x_axis_label)  # Change this label as per your data's X-axis
    plt.ylabel("NDCG@10")
    plt.grid(True)  # Add grid for better readability
    plt.tight_layout()  # Adjust layout to fit all elements neatly
    if legends:
        plt.legend()
    plt.savefig(output_filepath, dpi=300)  # Save with high resolution


def process_edges(src, dst, t, msg, memory, neighbor_loader):
    if src.nelement() > 0:
        # msg = msg.to(torch.float32)
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)


def train(
    train_loader,
    memory,
    gnn,
    node_pred,
    dataset,
    data,
    neighbor_loader,
    optimizer,
    criterion,
    assoc,
    eval_metric,
    evaluator,
):
    memory.train()
    gnn.train()
    node_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    label_t = dataset.get_label_time()  # check when does the first label start
    num_label_ts = 0
    total_score = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        query_t = batch.t[-1]
        # check if this batch moves to the next day
        if query_t > label_t:
            # find the node labels from the past day
            label_tuple = dataset.get_node_label(query_t)
            label_ts, label_srcs, labels = (
                label_tuple[0],
                label_tuple[1],
                label_tuple[2],
            )
            label_t = dataset.get_label_time()
            label_srcs = label_srcs.to(device)

            # Process all edges that are still in the past day
            previous_day_mask = batch.t < label_t
            process_edges(
                src[previous_day_mask],
                dst[previous_day_mask],
                t[previous_day_mask],
                msg[previous_day_mask],
                memory,
                neighbor_loader,
            )
            # Reset edges to be the edges from tomorrow so they can be used later
            src, dst, t, msg = (
                src[~previous_day_mask],
                dst[~previous_day_mask],
                t[~previous_day_mask],
                msg[~previous_day_mask],
            )

            """
            modified for node property prediction
            1. sample neighbors from the neighbor loader for all nodes to be predicted
            2. extract memory from the sampled neighbors and the nodes
            3. run gnn with the extracted memory embeddings and the corresponding time and message
            """
            n_id = label_srcs
            n_id_neighbors, mem_edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id_neighbors] = torch.arange(n_id_neighbors.size(0), device=device)

            z, last_update = memory(n_id_neighbors)

            z = gnn(
                z,
                last_update,
                mem_edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )
            z = z[assoc[n_id]]

            # loss and metric computation
            pred = node_pred(z)

            loss = criterion(pred, labels.to(device))
            np_pred = pred.cpu().detach().numpy()
            np_true = labels.cpu().detach().numpy()

            input_dict = {
                "y_true": np_true,
                "y_pred": np_pred,
                "eval_metric": [eval_metric],
            }
            result_dict = evaluator.eval(input_dict)
            score = result_dict[eval_metric]
            total_score += score
            num_label_ts += 1

            loss.backward()
            optimizer.step()
            total_loss += float(loss)

        # Update memory and neighbor loader with ground-truth state.
        process_edges(src, dst, t, msg, memory, neighbor_loader)
        memory.detach()

    metric_dict = {
        "ce": total_loss / num_label_ts,
    }
    metric_dict[eval_metric] = total_score / num_label_ts
    return metric_dict


@torch.no_grad()
def test(
    loader,
    memory,
    gnn,
    node_pred,
    dataset,
    data,
    neighbor_loader,
    assoc,
    eval_metric,
    evaluator,
):
    memory.eval()
    gnn.eval()
    node_pred.eval()
    total_score = 0
    label_t = dataset.get_label_time()  # check when does the first label start
    num_label_ts = 0

    for batch in loader:
        batch = batch.to(device)
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        query_t = batch.t[-1]
        if query_t > label_t:
            label_tuple = dataset.get_node_label(query_t)
            if label_tuple is None:
                break
            label_ts, label_srcs, labels = (
                label_tuple[0],
                label_tuple[1],
                label_tuple[2],
            )
            label_t = dataset.get_label_time()
            label_srcs = label_srcs.to(device)

            # Process all edges that are still in the past day
            previous_day_mask = batch.t < label_t
            process_edges(
                src[previous_day_mask],
                dst[previous_day_mask],
                t[previous_day_mask],
                msg[previous_day_mask],
                memory,
                neighbor_loader,
            )
            # Reset edges to be the edges from tomorrow so they can be used later
            src, dst, t, msg = (
                src[~previous_day_mask],
                dst[~previous_day_mask],
                t[~previous_day_mask],
                msg[~previous_day_mask],
            )

            """
            modified for node property prediction
            1. sample neighbors from the neighbor loader for all nodes to be predicted
            2. extract memory from the sampled neighbors and the nodes
            3. run gnn with the extracted memory embeddings and the corresponding time and message
            """
            n_id = label_srcs
            n_id_neighbors, mem_edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id_neighbors] = torch.arange(n_id_neighbors.size(0), device=device)

            z, last_update = memory(n_id_neighbors)
            z = gnn(
                z,
                last_update,
                mem_edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )
            z = z[assoc[n_id]]

            # loss and metric computation
            pred = node_pred(z)
            np_pred = pred.cpu().detach().numpy()
            np_true = labels.cpu().detach().numpy()

            input_dict = {
                "y_true": np_true,
                "y_pred": np_pred,
                "eval_metric": [eval_metric],
            }
            result_dict = evaluator.eval(input_dict)
            score = result_dict[eval_metric]
            total_score += score
            num_label_ts += 1

        process_edges(src, dst, t, msg, memory, neighbor_loader)

    metric_dict = {}
    metric_dict[eval_metric] = total_score / num_label_ts
    return metric_dict


def main(
    dataset,
    data,
    method,
    n_samples,
    train_mask,
    val_mask,
    test_mask,
    epochs=10,
    repeat_idx=1,
):
    eval_metric = dataset.eval_metric
    num_classes = dataset.num_classes

    evaluator = Evaluator(name=name)

    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]

    # Ensure to only sample actual destination nodes as negatives.
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

    train_loader = TemporalDataLoader(train_data, batch_size=args.train_batch_size)
    val_loader = TemporalDataLoader(val_data, batch_size=args.batch_size)
    test_loader = TemporalDataLoader(test_data, batch_size=args.batch_size)

    neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)

    memory_dim = time_dim = embedding_dim = 100

    memory = TGNMemory(
        data.num_nodes,
        data.msg.size(-1),
        memory_dim,
        time_dim,
        message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = (
        GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=data.msg.size(-1),
            time_enc=memory.time_enc,
        )
        .to(device)
        .float()
    )

    node_pred = NodePredictor(in_dim=embedding_dim, out_dim=num_classes).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(node_pred.parameters()),
        lr=lr,
    )

    criterion = torch.nn.CrossEntropyLoss()
    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    train_curve = []
    val_curve = []
    test_curve = []
    max_val_score = 0  # find the best test score based on validation score
    best_test_idx = 0
    for epoch in range(1, epochs + 1):
        start_time = timeit.default_timer()
        train_args = (
            memory,
            gnn,
            node_pred,
            dataset,
            data,
            neighbor_loader,
            optimizer,
            criterion,
            assoc,
            eval_metric,
            evaluator,
        )
        train_dict = train(train_loader, *train_args)
        print("------------------------------------")
        print(f"training Epoch: {epoch:02d}")
        print(train_dict)
        train_curve.append(train_dict[eval_metric])
        print(
            "Training takes--- %s seconds ---" % (timeit.default_timer() - start_time)
        )

        start_time = timeit.default_timer()
        val_dict = test(
            val_loader,
            memory,
            gnn,
            node_pred,
            dataset,
            data,
            neighbor_loader,
            assoc,
            eval_metric,
            evaluator,
        )
        print(val_dict)
        val_curve.append(val_dict[eval_metric])
        if val_dict[eval_metric] > max_val_score:
            max_val_score = val_dict[eval_metric]
            best_test_idx = epoch - 1
        print(
            "Validation takes--- %s seconds ---" % (timeit.default_timer() - start_time)
        )

        start_time = timeit.default_timer()
        test_dict = test(
            test_loader,
            memory,
            gnn,
            node_pred,
            dataset,
            data,
            neighbor_loader,
            assoc,
            eval_metric,
            evaluator,
        )
        print(test_dict)
        test_curve.append(test_dict[eval_metric])
        print("Test takes--- %s seconds ---" % (timeit.default_timer() - start_time))
        print("------------------------------------")
        dataset.reset_label_time()

    # code for plotting
    plot_curve(
        list(range(len(train_curve))),
        train_curve,
        fig_dirpath
        / f"repeat{repeat_idx}_{name}_al_tgn_train_curve_epoch{epochs}_{method}_{n_samples}samples.png",
    )
    plot_curve(
        list(range(len(val_curve))),
        val_curve,
        fig_dirpath
        / f"repeat{repeat_idx}_{name}_al_tgn_val_curve_epoch{epochs}_{method}_{n_samples}samples.png",
    )
    plot_curve(
        list(range(len(test_curve))),
        test_curve,
        fig_dirpath
        / f"repeat{repeat_idx}_{name}_al_tgn_test_curve_epoch{epochs}_{method}_{n_samples}samples.png",
    )

    max_test_score = test_curve[best_test_idx]
    print("------------------------------------")
    print("------------------------------------")
    print("best val score: ", max_val_score)
    print("best validation epoch   : ", best_test_idx + 1)
    print("best test score: ", max_test_score)

    tmp_results_fp = (
        Path("data/")
        / f"repeat{repeat_idx}_{name}_{method}_epochs{epochs}_{n_samples}samples.json"
    )
    tmp_results = {
        "train_scores": train_curve,
        "test_scores": test_curve,
        "val_scores": val_curve,
    }
    with open(tmp_results_fp, "w") as f:
        json.dump(tmp_results, f)

    return {
        "max_val_score": max_val_score,
        "max_test_score": max_test_score,
        "max_train_score": train_curve[best_test_idx],
    }, train_args


def estimate_model_change(
    remaining_train_loader,
    n_samples,
    memory,
    gnn,
    node_pred,
    dataset,
    data,
    neighbor_loader,
    optimizer,
    criterion,
    assoc,
    eval_metric,
    evaluator,
):
    memory.eval()
    gnn.eval()
    node_pred.eval()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    label_t = dataset.get_label_time()  # check when does the first label start
    num_label_ts = 0
    total_score = 0

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    # record expected model_changes, map back through edge mapping
    expected_model_changes_src = torch.tensor([])
    expected_model_changes_t = torch.tensor([])
    expected_model_changes_vals = torch.tensor([])
    all_e_id = torch.tensor([])

    for batch_idx, batch in enumerate(remaining_train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        query_t = batch.t[-1]

        # check if this batch moves to the next day
        if query_t > label_t:
            # find the node labels from the past day
            label_tuple = dataset.get_node_label(query_t)
            label_ts, label_srcs, labels = (
                label_tuple[0],
                label_tuple[1],
                label_tuple[2],
            )
            label_t = dataset.get_label_time()
            label_srcs = label_srcs.to(device)

            # Process all edges that are still in the past day
            previous_day_mask = batch.t < label_t
            process_edges(
                src[previous_day_mask],
                dst[previous_day_mask],
                t[previous_day_mask],
                msg[previous_day_mask],
                memory,
                neighbor_loader,
            )
            # Reset edges to be the edges from tomorrow so they can be used later
            src, dst, t, msg = (
                src[~previous_day_mask],
                dst[~previous_day_mask],
                t[~previous_day_mask],
                msg[~previous_day_mask],
            )

            """
            modified for node property prediction
            1. sample neighbors from the neighbor loader for all nodes to be predicted
            2. extract memory from the sampled neighbors and the nodes
            3. run gnn with the extracted memory embeddings and the corresponding time and message
            """
            n_id = label_srcs
            n_id_neighbors, mem_edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id_neighbors] = torch.arange(n_id_neighbors.size(0), device=device)

            z, last_update = memory(n_id_neighbors)
            z = gnn(
                z,
                last_update,
                mem_edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )
            z = z[assoc[n_id]]

            # loss and metric computation
            pred = node_pred(z)
            unreduced_losses = criterion(pred, labels.to(device))

            estimated_model_changes = torch.zeros(len(unreduced_losses))
            for j, loss in enumerate(unreduced_losses):
                # print(loss.size(), loss)
                loss.backward(retain_graph=True)
                total_loss += float(loss)

                # Calculate the sum of gradient magnitudes
                model_change = sum(
                    p.grad.norm().item()
                    for p in set(memory.parameters())
                    | set(gnn.parameters())
                    | set(node_pred.parameters())
                    if p.grad is not None
                )

                estimated_model_changes[j] = model_change

                # Reset gradients
                for p in (
                    set(memory.parameters())
                    | set(gnn.parameters())
                    | set(node_pred.parameters())
                ):
                    if p.grad is not None:
                        p.grad.zero_()

            expected_model_changes_src = torch.cat(
                [expected_model_changes_src, label_srcs.cpu()], dim=0
            )
            expected_model_changes_t = torch.cat(
                [expected_model_changes_t, label_ts.cpu()],
                dim=0,
            )
            expected_model_changes_vals = torch.cat(
                [expected_model_changes_vals, estimated_model_changes.cpu()], dim=0
            )
            all_e_id = torch.cat([all_e_id, e_id.cpu()], dim=0)

            num_label_ts += 1

        # Update memory and neighbor loader with ground-truth state.
        process_edges(src, dst, t, msg, memory, neighbor_loader)
        memory.detach()

    print("done processing all")

    # should sample pairs of (node, time) for active learning task
    return (
        expected_model_changes_src,
        expected_model_changes_t,
        expected_model_changes_vals,
        all_e_id,
    )


def get_sampled_mask_from_expected_model_change(
    i,
    train_indices,
    train_mask,
    sampled_indices,
    n_samples,
    data,
    train_args,
):
    remaining_train_indices = (
        train_indices
        if i == 0
        else train_indices[~torch.isin(train_indices, sampled_indices)]
    )
    remaining_train_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    remaining_train_mask[remaining_train_indices] = True

    # batches of size one here
    train_loader = TemporalDataLoader(
        data[remaining_train_mask], batch_size=args.batch_size
    )
    (
        src,
        t,
        model_change_vals,
        all_e_id,
    ) = estimate_model_change(
        train_loader, int(remaining_train_mask.sum()), *train_args
    )

    # all src and t which we computed loss over
    stacked = torch.stack([src, t], dim=1)
    stacked_map = {tuple(map(int, pair.tolist())): i for i, pair in enumerate(stacked)}

    # true in this case, does it generalize to all datasets?
    assert src.size(0) == t.size(0) == model_change_vals.size(0)

    # broadcast unique pairs to the entire set of remaining edges
    all_stacked = torch.stack(
        [data.src[remaining_train_mask], data.t[remaining_train_mask]], dim=1
    ).cpu()
    all_model_changes = torch.zeros_like(
        data.src[remaining_train_mask], dtype=model_change_vals.dtype
    )
    assert remaining_train_indices.size(0) == all_model_changes.size(0)

    # some combination of node & timestamp are cut out from the neighbors topk?
    for i, pair in enumerate(map(lambda x: tuple(map(int, x.tolist())), all_stacked)):
        if pair in stacked_map:
            all_model_changes[i] = model_change_vals[stacked_map[pair]]

    # retrieve a cumulative n_samles
    additional_samples = n_samples if i == 0 else n_samples - len(sampled_indices)
    intermediate_indices = torch.argsort(all_model_changes)[-additional_samples:].to(
        remaining_train_indices.device
    )
    print(
        "show highest expected model changes",
        all_model_changes[intermediate_indices[-10:]],
    )

    # map onto the train mask to get the sampled indices
    new_sampled_indices = remaining_train_indices[intermediate_indices]
    # handle previous tensor which was
    sampled_indices = torch.cat(
        [sampled_indices, new_sampled_indices],
        dim=0,
    )

    sampled_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    sampled_mask[sampled_indices] = True

    return sampled_mask, sampled_indices


def get_random_sampled_mask(i, train_indices, train_mask, sampled_indices, n_samples):
    # initialize the train mask to a random portion of the
    if i == 0:
        sampled_indices = torch.from_numpy(
            np.random.choice(train_indices, n_samples, replace=False)
        )
    else:
        new_samples_count = n_samples - len(sampled_indices)

        used_indices = torch.isin(train_indices, sampled_indices)
        available_indices = train_indices[~used_indices]
        if new_samples_count > len(available_indices):
            raise ValueError(
                "Not enough remaining samples to meet the required n_samples."
            )

        new_sampled_indices = torch.from_numpy(
            np.random.choice(available_indices, new_samples_count, replace=False)
        )
        sampled_indices = torch.cat((sampled_indices, new_sampled_indices), dim=0)

    # create mask using previous steps
    sampled_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    sampled_mask[sampled_indices] = True

    return sampled_mask, sampled_indices


def get_sampled_mask_from_kcenter_greedy(
    i,
    train_indices,
    train_mask,
    sampled_indices,
    n_samples,
    data,
    train_args,
):
    remaining_train_mask, remaining_train_indices = nprop_utils.get_remaining_portion(
        i, train_indices, train_mask, sampled_indices
    )
    train_loader = TemporalDataLoader(
        data[remaining_train_mask], batch_size=args.batch_size
    )

    with torch.no_grad():
        src, t, node_embeddings = nprop_utils.get_node_embeddings(
            train_loader, int(remaining_train_mask.sum()), *train_args
        )
    print("done computing all node embeddings")
    stacked = torch.stack([src, t], dim=1)
    stacked_map = {tuple(map(int, pair.tolist())): i for i, pair in enumerate(stacked)}

    # get a N x N matrix covering all nodes included in the dataset
    dist_mat = np.matmul(node_embeddings.numpy(), node_embeddings.numpy().transpose())
    sq = np.array(dist_mat.diagonal()).reshape(len(src), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)

    # TODO: do not filter the remaining training mask, since want to include
    # all remaining training edges
    # the embeddings were computed over all the non-labelled, remaining edges
    remaining_nodes = torch.zeros_like(src, dtype=torch.bool)
    sorted_node_idx = torch.zeros_like(src, dtype=torch.long)
    mat = dist_mat  # [~remaining_nodes, :][:, remaining_nodes]

    # get a mask for all outstanding nodes
    n = len(src)
    for i in tqdm(range(n), ncols=100):
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = np.arange(len(src))[~remaining_nodes][q_idx_]
        remaining_nodes[q_idx] = True
        mat = np.delete(mat, q_idx_, 0)
        mat = np.append(mat, dist_mat[~remaining_nodes, q_idx][:, None], axis=1)
        sorted_node_idx[i] = q_idx

    # collapse back to only include non-zero values
    # sorted_node_idx = sorted_node_idx[sorted_node_idx.nonzero().squeeze()]

    # invert the relation, to be able to retrieve the rank with the node idx
    inv_sorted_node_idx = torch.zeros_like(sorted_node_idx, dtype=torch.long)
    inv_sorted_node_idx[sorted_node_idx] = torch.arange(
        sorted_node_idx.size(0), dtype=torch.long
    )

    # broadcasting the rank of the nodes to their edges
    all_stacked = torch.stack(
        [data.src[remaining_train_mask], data.t[remaining_train_mask]], dim=1
    ).cpu()
    all_sorted_node_idx = torch.zeros_like(
        data.src[remaining_train_mask], dtype=sorted_node_idx.dtype
    )
    for i, pair in enumerate(map(lambda x: tuple(map(int, x.tolist())), all_stacked)):
        if pair in stacked_map:
            all_sorted_node_idx[i] = inv_sorted_node_idx[stacked_map[pair]]

    additional_samples = n_samples if i == 0 else n_samples - len(sampled_indices)
    intermediate_indices = torch.argsort(all_sorted_node_idx)[:additional_samples].to(
        remaining_train_indices.device
    )
    print(
        "show edges for highest kcenter greedy score nodes",
        all_sorted_node_idx[intermediate_indices[:10]],
    )

    # map onto the train mask to get the sampled indices
    new_sampled_indices = remaining_train_indices[intermediate_indices]

    # handle previous tensor which was
    sampled_indices = torch.cat(
        [sampled_indices, new_sampled_indices],
        dim=0,
    )

    sampled_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    sampled_mask[sampled_indices] = True

    return sampled_mask, sampled_indices


def execute_training(method, n_samples_range, epochs, repeat_idx):
    sampled_indices = None
    data_keys = ["max_test_score", "max_val_score", "max_train_score"]
    results = {method: {key: [] for key in data_keys}}
    for i, n_samples in enumerate(n_samples_range):
        # reload data on each pass
        dataset = PyGNodePropPredDataset(name=name, root="datasets")
        train_mask = dataset.train_mask
        val_mask = dataset.val_mask
        test_mask = dataset.test_mask

        data = dataset.get_TemporalData()
        data = data.to(device)
        train_indices = torch.where(train_mask)[0]

        print(f"training model with {n_samples} training samples")

        if i == 0 or method == "random":
            print(
                f"repeat ({repeat_idx+1}/{args.repeats}) ({i+1}/{len(n_samples_range)}) retrieving training samples randomly"
            )
            sampled_mask, sampled_indices = get_random_sampled_mask(
                i, train_indices, train_mask, sampled_indices, n_samples
            )
        elif method == "expected_model_change":
            assert train_args, "no training args setup"
            print(
                f"repeat ({repeat_idx+1}/{args.repeats}) ({i+1}/{len(n_samples_range)}) retrieving training samples with expected model change"
            )
            (
                sampled_mask,
                sampled_indices,
            ) = get_sampled_mask_from_expected_model_change(
                i,
                train_indices,
                train_mask,
                sampled_indices,
                n_samples,
                data,
                train_args,
            )
        elif method == "kcenter_greedy":
            assert train_args, "no training args setup"
            print(
                f"repeat ({repeat_idx+1}/{args.repeats}) ({i+1}/{len(n_samples_range)}) retrieving training samples with expected model change"
            )
            (
                sampled_mask,
                sampled_indices,
            ) = get_sampled_mask_from_kcenter_greedy(
                i,
                train_indices,
                train_mask,
                sampled_indices,
                n_samples,
                data,
                train_args,
            )
        else:
            raise ValueError("No such method implemented")

        print(
            f"repeat ({repeat_idx+1}/{args.repeats}) ({i+1}/{len(n_samples_range)}) using {sampled_mask.sum()} training samples"
        )
        _results, train_args = main(
            dataset,
            data,
            method,
            n_samples,
            sampled_mask,
            val_mask,
            test_mask,
            epochs=epochs,
        )
        for key in _results:
            results[method][key].append(_results[key])

    return results


def get_executor_pool(use_thread_pool: bool = False, max_workers: int = 12):
    global _AS_COMPLETED_FN

    if use_thread_pool:
        pool_executor_cls = futures.ThreadPoolExecutor
    else:
        pool_executor_cls = futures.ProcessPoolExecutor

    executor_kwargs = dict(max_workers=max_workers)
    _AS_COMPLETED_FN = futures.as_completed

    return pool_executor_cls(**executor_kwargs)


def get_futures(executor: futures.ProcessPoolExecutor, fn, kwargs_list: list):
    return {executor.submit(fn, **kwargs): kwargs for kwargs in kwargs_list}


def complete_futures(futures: dict):
    results = []
    for future in _AS_COMPLETED_FN(futures):
        result = future.result()
        results.append(result)
    return results


def execute_parallel(fn, kwargs_list: list, worker_count: int = 12):
    executor = get_executor_pool(max_workers=worker_count)
    futures = get_futures(executor, fn, kwargs_list)
    return complete_futures(futures)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    methods = ["random", "expected_model_change", "kcenter_greedy"]
    n_samples_range = list(
        np.linspace(args.min_samples, args.max_samples, args.count).astype(int)
    )
    data_keys = ["max_test_score", "max_val_score", "max_train_score"]
    print("running for methods", methods)
    print("samples range", n_samples_range)

    kwargs_list = []
    for method in methods:
        for repeat_idx in range(args.repeats):
            kwargs_list.append(
                {
                    "method": method,
                    "epochs": args.epochs,
                    "n_samples_range": n_samples_range,
                    "repeat_idx": repeat_idx,
                }
            )

    results = execute_parallel(
        execute_training, kwargs_list, worker_count=len(kwargs_list)
    )

    #for key in data_keys:
    #    fp = (
    #        fig_dirpath
    #        / f"{'_'.join(methods)}_{args.repeats}repeats_{args.count}nsamples_{args.epochs}epochs_{name}_{key}.png"
    #    )
    #    print(f"plotting {fp}")
    #    data = [results[method][key] for method in results]
    #    plot_curve(
    #        n_samples_range,
    #        data,
    #        fp,
    #        x_axis_label="Training Sample Count",
    #        legends=methods,
    #    )

    data_path = Path("data/")
    data_path.mkdir(exist_ok=True, parents=True)
    results_path = (
        data_path
        / f"{'_'.join(methods)}_{args.count}nsamples_{args.epochs}epochs_{name}_results.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f)
