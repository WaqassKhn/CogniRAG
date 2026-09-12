"""
tests/test_graph_visualizer.py - Unit tests for GraphVisualizer.
"""

import pytest
from rag.graph.visualizer import GraphVisualizer


def test_visualizer_node_colors():
    vis = GraphVisualizer()

    pol_color = vis.get_node_color("POLICY")
    assert pol_color["background"] == "#10b981"

    role_color = vis.get_node_color("ROLE")
    assert role_color["background"] == "#f59e0b"

    dept_color = vis.get_node_color("DEPARTMENT")
    assert dept_color["background"] == "#06b6d4"

    def_color = vis.get_node_color("CUSTOM_UNKNOWN")
    assert def_color["background"] == "#9ca3af"


def test_visualizer_empty_state():
    vis = GraphVisualizer()

    empty_html = vis.render_empty_state_html("Custom empty message")
    assert "Knowledge Graph Visualizer" in empty_html
    assert "Custom empty message" in empty_html

    no_nodes_html = vis.generate_network_html(nodes=[], edges=[])
    assert "No entity nodes found" in no_nodes_html


def test_visualizer_network_generation():
    vis = GraphVisualizer()

    nodes = [
        {"id": "Travel Policy", "label": "Travel Policy", "type": "POLICY", "description": "Travel rules"},
        {"id": "VP Operations", "label": "VP Operations", "type": "ROLE", "description": "Operations Head"},
    ]
    edges = [
        {
            "source": "Travel Policy",
            "target": "VP Operations",
            "type": "REQUIRES_APPROVAL_FROM",
            "description": "Approval threshold $5,000",
            "chunk_id": "chunk_1",
        }
    ]

    html = vis.generate_network_html(nodes=nodes, edges=edges, physics=False)

    assert "<html" in html.lower() or "<div" in html.lower()
    assert "Travel Policy" in html
    assert "VP Operations" in html
