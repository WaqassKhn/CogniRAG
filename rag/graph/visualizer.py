"""
rag/graph/visualizer.py - Interactive Knowledge Graph Network Visualizer for CogniRAG.

Generates interactive PyVis network graphs with custom dark aesthetics matching
CogniRAG's minimalist design system (#080c0a background, emerald/amber/cyan node types,
physics simulation, and responsive hover cards).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    Network = None
    PYVIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class GraphVisualizer:
    """
    Builds interactive PyVis network HTML graphs for Knowledge Graph exploration in Streamlit.
    """

    # Dark minimal color palette keyed by enterprise entity types
    TYPE_COLORS = {
        "POLICY": {"background": "#10b981", "border": "#059669"},       # Emerald
        "ROLE": {"background": "#f59e0b", "border": "#d97706"},         # Amber
        "DEPARTMENT": {"background": "#06b6d4", "border": "#0891b2"},   # Cyan
        "METRIC": {"background": "#a855f7", "border": "#7e22ce"},       # Purple
        "ORGANIZATION": {"background": "#3b82f6", "border": "#2563eb"}, # Blue
        "FISCAL_PERIOD": {"background": "#ec4899", "border": "#db2777"},# Pink
        "DEFAULT": {"background": "#9ca3af", "border": "#6b7280"},      # Gray
    }

    def __init__(
        self,
        bg_color: str = "#080c0a",
        height: str = "620px",
        width: str = "100%",
    ):
        self.bg_color = bg_color
        self.height = height
        self.width = width

    def get_node_color(self, entity_type: str) -> Dict[str, str]:
        """Returns background and border hex colors for a given entity type."""
        norm_type = entity_type.upper().strip() if entity_type else "DEFAULT"
        return self.TYPE_COLORS.get(norm_type, self.TYPE_COLORS["DEFAULT"])

    def generate_network_html(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        physics: bool = True,
    ) -> str:
        """
        Builds a PyVis network graph and returns the full HTML document string.
        """
        if not PYVIS_AVAILABLE or Network is None:
            return self.render_empty_state_html(
                "PyVis is not installed. Please install `pyvis` to enable interactive graph visualization."
            )

        if not nodes:
            return self.render_empty_state_html(
                "No entity nodes found in the knowledge graph. Upload enterprise documents to generate graph relationships."
            )

        try:
            net = Network(
                height=self.height,
                width=self.width,
                bgcolor=self.bg_color,
                font_color="#f3f4f6",
                directed=True,
                notebook=False,
            )

            # Add nodes with type-specific styling and hover titles
            for n in nodes:
                node_id = n["id"]
                label = n.get("label", node_id)
                etype = n.get("type", "ENTITY")
                desc = n.get("description", "")
                colors = self.get_node_color(etype)

                title_html = f"<b>{label}</b><br><i>Type:</i> {etype}"
                if desc:
                    title_html += f"<br><i>Details:</i> {desc}"

                # Size nodes based on label importance or degree
                size = 22 if etype in ("POLICY", "ORGANIZATION") else 17

                net.add_node(
                    node_id,
                    label=label,
                    title=title_html,
                    color={
                        "background": colors["background"],
                        "border": colors["border"],
                        "highlight": {"background": "#34d399", "border": "#10b981"},
                        "hover": {"background": "#6ee7b7", "border": "#059669"},
                    },
                    size=size,
                    shape="dot",
                    font={
                        "color": "#f3f4f6",
                        "size": 13,
                        "face": "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
                    },
                    borderWidth=2,
                )

            # Add relationship edges with labels and provenance
            for e in edges:
                src = e["source"]
                tgt = e["target"]
                rel_type = e.get("type", "RELATES_TO")
                desc = e.get("description", "")
                chunk_id = e.get("chunk_id", "")

                edge_title = f"Relation: <b>{rel_type}</b>"
                if desc:
                    edge_title += f"<br>Note: {desc}"
                if chunk_id:
                    edge_title += f"<br>Evidence: <code>{chunk_id}</code>"

                net.add_edge(
                    src,
                    tgt,
                    title=edge_title,
                    label=rel_type.replace("_", " ").lower(),
                    color={"color": "rgba(255, 255, 255, 0.28)", "highlight": "#10b981", "hover": "#34d399"},
                    arrows="to",
                    font={"color": "#9ca3af", "size": 10, "align": "middle"},
                    smooth={"type": "continuous"},
                )

            # Physics options
            if physics:
                net.barnes_hut(
                    gravity=-2500,
                    central_gravity=0.3,
                    spring_length=130,
                    spring_strength=0.04,
                    damping=0.09,
                    overlap=0,
                )
            else:
                net.toggle_physics(False)

            # Save and read back HTML safely across OS environments
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tf:
                temp_path = tf.name

            net.save_graph(temp_path)

            with open(temp_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            Path(temp_path).unlink(missing_ok=True)
            return html_content

        except Exception as exc:
            logger.error(f"[GraphVisualizer] Failed to generate PyVis HTML: {exc}", exc_info=True)
            return self.render_empty_state_html(f"Error compiling graph visualization: {exc}")

    def render_empty_state_html(self, message: str) -> str:
        """Renders a styled dark-theme placeholder card when the graph is empty or offline."""
        return f"""
        <div style="
            background-color: #0c130f;
            border: 1px solid #18261e;
            border-radius: 8px;
            padding: 50px 20px;
            text-align: center;
            color: #9ca3af;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Inter, sans-serif;
            margin: 10px 0;
        ">
            <div style="font-size: 2.2rem; margin-bottom: 12px;">🕸️</div>
            <div style="font-size: 1.15rem; color: #f3f4f6; font-weight: 600; margin-bottom: 8px;">
                Knowledge Graph Visualizer
            </div>
            <div style="font-size: 0.9rem; color: #9ca3af; max-width: 500px; margin: 0 auto; line-height: 1.5;">
                {message}
            </div>
        </div>
        """
