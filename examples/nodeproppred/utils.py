import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_remaining_portion(i, train_indices, train_mask, sampled_indices):
    remaining_train_indices = (
        train_indices
        if i == 0
        else train_indices[~torch.isin(train_indices, sampled_indices)]
    )
    remaining_train_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    remaining_train_mask[remaining_train_indices] = True

    return remaining_train_mask, remaining_train_indices


def process_edges(src, dst, t, msg, memory, neighbor_loader):
    if src.nelement() > 0:
        # msg = msg.to(torch.float32)
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)


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
    plt.ylabel("Score")
    plt.grid(True)  # Add grid for better readability
    plt.tight_layout()  # Adjust layout to fit all elements neatly
    if legends:
        plt.legend()
    plt.savefig(output_filepath, dpi=300)  # Save with high resolution


def get_node_embeddings(
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

    expected_model_changes_src = torch.tensor([])
    expected_model_changes_t = torch.tensor([])
    all_node_embeddings = torch.tensor([])

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

            with torch.no_grad():
                z, last_update = memory(n_id_neighbors)
                z = gnn(
                    z,
                    last_update,
                    mem_edge_index,
                    data.t[e_id].to(device),
                    data.msg[e_id].to(device),
                )
                z = z[assoc[n_id]]

            all_node_embeddings = torch.cat((all_node_embeddings, z.cpu()), dim=0)
            expected_model_changes_src = torch.cat(
                [expected_model_changes_src, label_srcs.cpu()], dim=0
            )
            expected_model_changes_t = torch.cat(
                [expected_model_changes_t, label_ts.cpu()],
                dim=0,
            )

            num_label_ts += 1

        # Update memory and neighbor loader with ground-truth state.
        process_edges(src, dst, t, msg, memory, neighbor_loader)
        memory.detach()

    # should sample pairs of (node, time) for active learning task
    return (
        expected_model_changes_src,
        expected_model_changes_t,
        all_node_embeddings,
    )
