"""
NOVELTY 2: Priority Aware Risk Propagation
Graph-based spatial modeling where risk propagates between neighboring zones
"""
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PROPAGATION_DECAY, MAX_PROPAGATION_HOPS
from data_engineering.h3_processor import H3Processor


class PriorityAwareRiskPropagation:
    """
    NOVELTY 2: Priority Aware Risk Propagation
    
    Graph-based spatial modeling where:
    - Each H3 hexagon is a node
    - Neighboring hexagons share risk
    - Risk propagates with decay
    
    Formula: Final_Risk = own_risk + Σ(neighbor_influence * decay^distance)
    
    This creates a more realistic citywide risk map by considering
    spatial dependencies between zones.
    """
    
    def __init__(self, decay_factor=PROPAGATION_DECAY, max_hops=MAX_PROPAGATION_HOPS):
        """
        Initialize risk propagation system
        
        Args:
            decay_factor: How much risk decays per hop (0-1)
            max_hops: Maximum propagation distance
        """
        self.decay_factor = decay_factor
        self.max_hops = max_hops
        self.h3_processor = H3Processor()
        self.graph = None
        self.zone_risks = {}
        
    def build_graph(self, h3_indices):
        """
        Build graph from H3 indices
        
        Args:
            h3_indices: Collection of H3 index strings
            
        Returns:
            NetworkX graph
        """
        self.graph = nx.Graph()
        h3_set = set(h3_indices)
        
        # Add nodes
        for h3_idx in h3_set:
            self.graph.add_node(h3_idx)
        
        # Add edges between neighbors
        for h3_idx in h3_set:
            neighbors = self.h3_processor.get_neighbors(h3_idx, ring_size=1)
            for neighbor in neighbors:
                if neighbor in h3_set and neighbor != h3_idx:
                    self.graph.add_edge(h3_idx, neighbor, weight=1)
        
        print(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def set_zone_risks(self, risk_dict):
        """
        Set initial risk values for zones
        
        Args:
            risk_dict: Dictionary mapping H3 index to risk score
        """
        self.zone_risks = risk_dict.copy()
    
    def propagate_risk(self, h3_indices=None, risk_values=None):
        """
        Propagate risk across the graph
        
        Args:
            h3_indices: Optional list of H3 indices (uses graph nodes if None)
            risk_values: Optional dict of initial risks (uses zone_risks if None)
            
        Returns:
            Dictionary mapping H3 index to propagated risk
        """
        if self.graph is None:
            if h3_indices is None:
                raise ValueError("No graph built. Provide h3_indices or call build_graph first.")
            self.build_graph(h3_indices)
        
        if risk_values is not None:
            self.zone_risks = risk_values
        
        propagated_risks = {}
        
        for node in self.graph.nodes():
            # Start with own risk
            own_risk = self.zone_risks.get(node, 0)
            
            # Calculate neighbor influence
            neighbor_influence = self._calculate_neighbor_influence(node)
            
            # Combine own risk and neighbor influence
            # Own risk weighted more heavily
            final_risk = 0.7 * own_risk + 0.3 * neighbor_influence
            
            propagated_risks[node] = np.clip(final_risk, 0, 1)
        
        return propagated_risks
    
    def _calculate_neighbor_influence(self, node):
        """
        Calculate total influence from neighbors
        
        Args:
            node: H3 index of target node
            
        Returns:
            Weighted sum of neighbor risks
        """
        total_influence = 0
        total_weight = 0
        
        for hop in range(1, self.max_hops + 1):
            # Get nodes at this distance
            try:
                neighbors_at_hop = self._get_nodes_at_distance(node, hop)
            except:
                continue
            
            # Calculate decay for this hop
            decay = self.decay_factor ** hop
            
            for neighbor in neighbors_at_hop:
                if neighbor in self.zone_risks:
                    neighbor_risk = self.zone_risks[neighbor]
                    total_influence += neighbor_risk * decay
                    total_weight += decay
        
        if total_weight > 0:
            return total_influence / total_weight
        return 0
    
    def _get_nodes_at_distance(self, node, distance):
        """Get nodes at exactly the specified distance"""
        if distance == 1:
            return set(self.graph.neighbors(node))
        
        # Use BFS to find nodes at exact distance
        nodes_at_distance = set()
        visited = {node}
        current_level = {node}
        
        for d in range(distance):
            next_level = set()
            for n in current_level:
                for neighbor in self.graph.neighbors(n):
                    if neighbor not in visited:
                        next_level.add(neighbor)
                        visited.add(neighbor)
            current_level = next_level
        
        return current_level
    
    def calculate_propagated_scores(self, gold_df, risk_col='base_risk_score'):
        """
        Calculate propagated risk scores for gold layer data
        
        Args:
            gold_df: Gold layer DataFrame with H3 indices and risk scores
            risk_col: Name of risk score column
            
        Returns:
            DataFrame with propagated risk scores
        """
        if gold_df.empty:
            return gold_df
        
        df = gold_df.copy()
        
        # Build graph from H3 indices
        h3_indices = df['h3_index'].dropna().unique()
        self.build_graph(h3_indices)
        
        # Set initial risks
        risk_dict = dict(zip(df['h3_index'], df[risk_col]))
        self.set_zone_risks(risk_dict)
        
        # Propagate
        propagated = self.propagate_risk()
        
        # Add to DataFrame
        df['propagated_risk'] = df['h3_index'].map(propagated)
        
        # Calculate risk change
        df['risk_change'] = df['propagated_risk'] - df[risk_col]
        df['risk_amplified'] = df['risk_change'] > 0.05
        
        # Update risk category
        df['propagated_risk_category'] = pd.cut(
            df['propagated_risk'],
            bins=[-np.inf, 0.3, 0.6, 0.8, np.inf],
            labels=['low', 'medium', 'high', 'critical']
        )
        
        return df
    
    def identify_risk_clusters(self, min_cluster_size=3, risk_threshold=0.6):
        """
        Identify clusters of high-risk zones
        
        Args:
            min_cluster_size: Minimum zones to form a cluster
            risk_threshold: Minimum risk to be considered high-risk
            
        Returns:
            List of clusters (each cluster is a set of H3 indices)
        """
        if self.graph is None:
            return []
        
        # Get high-risk nodes
        high_risk_nodes = {
            node for node, risk in self.zone_risks.items()
            if risk >= risk_threshold
        }
        
        # Find connected components among high-risk nodes
        subgraph = self.graph.subgraph(high_risk_nodes)
        clusters = list(nx.connected_components(subgraph))
        
        # Filter by minimum size
        clusters = [c for c in clusters if len(c) >= min_cluster_size]
        
        return clusters
    
    def get_hotspot_zones(self, top_n=10):
        """
        Get top N hotspot zones based on propagated risk
        
        Args:
            top_n: Number of top zones to return
            
        Returns:
            List of (h3_index, risk_score) tuples
        """
        propagated = self.propagate_risk()
        sorted_zones = sorted(propagated.items(), key=lambda x: x[1], reverse=True)
        return sorted_zones[:top_n]
    
    def calculate_zone_influence(self, h3_index):
        """
        Calculate how much a zone influences its neighbors
        
        Args:
            h3_index: H3 index of zone
            
        Returns:
            Dictionary with influence metrics
        """
        if self.graph is None or h3_index not in self.graph:
            return {}
        
        own_risk = self.zone_risks.get(h3_index, 0)
        neighbors = list(self.graph.neighbors(h3_index))
        
        # Calculate influence on each neighbor
        influences = {}
        for neighbor in neighbors:
            neighbor_risk = self.zone_risks.get(neighbor, 0)
            influence = own_risk * self.decay_factor
            influences[neighbor] = {
                'neighbor_original_risk': neighbor_risk,
                'influence_added': influence,
                'potential_new_risk': neighbor_risk + influence * 0.3
            }
        
        return {
            'zone': h3_index,
            'own_risk': own_risk,
            'neighbor_count': len(neighbors),
            'total_influence': sum(i['influence_added'] for i in influences.values()),
            'neighbor_influences': influences
        }
    
    def get_propagation_summary(self):
        """
        Get summary of risk propagation effects
        
        Returns:
            Dictionary with propagation statistics
        """
        if not self.zone_risks:
            return {}
        
        original_risks = list(self.zone_risks.values())
        propagated = self.propagate_risk()
        propagated_risks = list(propagated.values())
        
        return {
            'total_zones': len(self.zone_risks),
            'original_mean_risk': np.mean(original_risks),
            'propagated_mean_risk': np.mean(propagated_risks),
            'original_max_risk': max(original_risks),
            'propagated_max_risk': max(propagated_risks),
            'zones_amplified': sum(1 for h3 in propagated 
                                   if propagated[h3] > self.zone_risks.get(h3, 0) + 0.05),
            'zones_dampened': sum(1 for h3 in propagated 
                                  if propagated[h3] < self.zone_risks.get(h3, 0) - 0.05)
        }
    
    def visualize_graph_data(self):
        """
        Get graph data for visualization
        
        Returns:
            Dictionary with nodes and edges for visualization
        """
        if self.graph is None:
            return {'nodes': [], 'edges': []}
        
        propagated = self.propagate_risk()
        
        nodes = []
        for node in self.graph.nodes():
            center = self.h3_processor.h3_to_center(node)
            nodes.append({
                'id': node,
                'lat': center[0],
                'lon': center[1],
                'original_risk': self.zone_risks.get(node, 0),
                'propagated_risk': propagated.get(node, 0)
            })
        
        edges = []
        for u, v in self.graph.edges():
            edges.append({
                'source': u,
                'target': v
            })
        
        return {'nodes': nodes, 'edges': edges}


def test_risk_propagation():
    """Test risk propagation functionality"""
    print("Testing Priority Aware Risk Propagation...")
    print("=" * 70)
    
    # Create sample H3 indices (NYC area)
    h3_proc = H3Processor()
    
    # Generate sample zones around NYC
    center_h3 = h3_proc.lat_lon_to_h3(40.7128, -74.0060)
    h3_indices = list(h3_proc.get_neighbors(center_h3, ring_size=3))
    
    print(f"\n1. Created {len(h3_indices)} H3 zones")
    
    # Create sample risk values
    np.random.seed(42)
    risk_values = {h3: np.random.uniform(0, 1) for h3 in h3_indices}
    
    # Set some high-risk zones
    high_risk_zones = list(h3_indices)[:5]
    for h3 in high_risk_zones:
        risk_values[h3] = np.random.uniform(0.7, 1.0)
    
    print(f"   Set {len(high_risk_zones)} high-risk zones")
    
    # Initialize propagation
    propagator = PriorityAwareRiskPropagation(decay_factor=0.5, max_hops=2)
    propagator.build_graph(h3_indices)
    propagator.set_zone_risks(risk_values)
    
    # Propagate risk
    print("\n2. Propagating risk...")
    propagated = propagator.propagate_risk()
    
    # Summary
    summary = propagator.get_propagation_summary()
    print(f"\n3. Propagation Summary:")
    print(f"   Total zones: {summary['total_zones']}")
    print(f"   Original mean risk: {summary['original_mean_risk']:.3f}")
    print(f"   Propagated mean risk: {summary['propagated_mean_risk']:.3f}")
    print(f"   Zones amplified: {summary['zones_amplified']}")
    print(f"   Zones dampened: {summary['zones_dampened']}")
    
    # Hotspots
    print("\n4. Top 5 Hotspot Zones:")
    hotspots = propagator.get_hotspot_zones(top_n=5)
    for h3, risk in hotspots:
        original = risk_values.get(h3, 0)
        print(f"   {h3[:12]}...: {risk:.3f} (original: {original:.3f})")
    
    # Clusters
    print("\n5. Risk Clusters:")
    clusters = propagator.identify_risk_clusters(min_cluster_size=2, risk_threshold=0.5)
    print(f"   Found {len(clusters)} high-risk clusters")
    for i, cluster in enumerate(clusters[:3]):
        print(f"   Cluster {i+1}: {len(cluster)} zones")
    
    # Zone influence
    print("\n6. Sample Zone Influence:")
    sample_zone = high_risk_zones[0]
    influence = propagator.calculate_zone_influence(sample_zone)
    print(f"   Zone: {sample_zone[:12]}...")
    print(f"   Own risk: {influence['own_risk']:.3f}")
    print(f"   Neighbors: {influence['neighbor_count']}")
    print(f"   Total influence: {influence['total_influence']:.3f}")
    
    print("\n" + "=" * 70)
    print("Risk Propagation Test Complete!")
    
    return propagator


if __name__ == "__main__":
    test_risk_propagation()
