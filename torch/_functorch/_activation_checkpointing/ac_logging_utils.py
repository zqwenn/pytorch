import json
import logging
from typing import List
from torch._logging import trace_structured
from torch.fx import Graph, Node

log: logging.Logger = logging.getLogger(__name__)

def create_joint_graph_node_information(joint_graph, recomputable_node_info):
    joint_graph_node_information = {}
    for i, joint_graph_node in enumerate(joint_graph.nodes):
        is_recomputable_candidate = joint_graph_node.name in recomputable_node_info
        tensor_meta = joint_graph_node.meta.get("tensor_meta")
        shape = getattr(tensor_meta, 'shape', []) if tensor_meta else []

        node_info = {
            "index": i,
            "name": joint_graph_node.name,
            "is_recomputable_candidate": is_recomputable_candidate,
            "target": str(joint_graph_node.target),
            "shape": str(shape),
            "input_arguments": [inp.name for inp in joint_graph_node.all_input_nodes],
            "stack_trace": joint_graph_node.meta.get("stack_trace", ""),
        }

        if is_recomputable_candidate:
            idx = recomputable_node_info[joint_graph_node.name]
            node_info["recomputable_candidate_info"] = {
                "memory": None,
                "runtime": None,
                "recomputable_node_idx": idx,
                "is_saved": False,
                "is_recomputed": False,
            }

        joint_graph_node_information[joint_graph_node.name] = node_info
    return joint_graph_node_information


def create_joint_graph_edges(joint_graph):
    joint_graph_edges = [
        (inp.name, node.name)
        for node in joint_graph.nodes
        for inp in node.all_input_nodes
    ]
    return joint_graph_edges


def create_activation_checkpointing_logging_structure_payload(
    joint_graph,
    joint_graph_node_information,
    joint_graph_edges,
    all_recomputable_banned_nodes,
    expected_runtime,
    saved_node_idxs,
    recomputable_node_idxs,
    memories_banned_nodes,
    runtimes_banned_nodes,
    min_cut_saved_values,
):
    activation_checkpointing_logging_structure_payload = {
        "Joint Graph Size": len(joint_graph.nodes),
        "Joint Graph Edges": {
            "Total": len(joint_graph_edges),
            "Edges": joint_graph_edges
        },
        "Joint Graph Node Information": joint_graph_node_information,
        "Recomputable Banned Nodes Order": [node.name for node in all_recomputable_banned_nodes],
        "Expected Runtime": expected_runtime,
        "Knapsack Saved Nodes": saved_node_idxs,
        "Knapsack Recomputed Nodes": recomputable_node_idxs,
        "Knapsack Input Memories": memories_banned_nodes,
        "Knapsack Input Runtimes": runtimes_banned_nodes,
        "Min Cut Solution Saved Values": [node.name for node in min_cut_saved_values],
    }
    return activation_checkpointing_logging_structure_payload


def create_structured_trace_for_min_cut_info(
    joint_graph,
    all_recomputable_banned_nodes,
    saved_node_idxs,
    recomputable_node_idxs,
    expected_runtime,
    memory_budget_used,
    memories_banned_nodes,
    runtimes_banned_nodes,
    min_cut_saved_values,
):
    recomputable_node_info = {node.name: idx for idx, node in enumerate(all_recomputable_banned_nodes)}
    joint_graph_node_information = create_joint_graph_node_information(joint_graph, recomputable_node_info)

    for node_name, node_info in joint_graph_node_information.items():
        if node_info["is_recomputable_candidate"]:
            idx = recomputable_node_info[node_name]
            node_info["recomputable_candidate_info"]["memory"] = memories_banned_nodes[idx]
            node_info["recomputable_candidate_info"]["runtime"] = runtimes_banned_nodes[idx]
            node_info["recomputable_candidate_info"]["is_saved"] = idx in saved_node_idxs
            node_info["recomputable_candidate_info"]["is_recomputed"] = idx in recomputable_node_idxs

    joint_graph_edges = create_joint_graph_edges(joint_graph)
    activation_checkpointing_logging_structure_payload = create_activation_checkpointing_logging_structure_payload(
        joint_graph,
        joint_graph_node_information,
        joint_graph_edges,
        all_recomputable_banned_nodes,
        expected_runtime,
        saved_node_idxs,
        recomputable_node_idxs,
        memories_banned_nodes,
        runtimes_banned_nodes,
        min_cut_saved_values,
    )

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "min_cut_information",
            "encoding": "json",
        },
        payload_fn=lambda: json.dumps(activation_checkpointing_logging_structure_payload),
    )
