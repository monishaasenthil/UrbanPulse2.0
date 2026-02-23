"""
Emergency Router - Priority Vehicle Routing
Generates optimal routes avoiding high-risk zones
"""
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import OUTPUTS_DIR
from data_engineering.h3_processor import H3Processor


class EmergencyRouter:
    """
    Emergency Vehicle Routing System
    
    Generates optimal routes for emergency vehicles by:
    - Avoiding high-risk zones
    - Minimizing response time
    - Considering real-time traffic
    """
    
    def __init__(self):
        self.h3_processor = H3Processor()
        self.graph = None
        self.zone_risks = {}
        self.routing_history = []
        
    def build_routing_graph(self, zone_data, risk_col='propagated_risk'):
        """
        Build routing graph from zone data
        
        Args:
            zone_data: DataFrame with zone information
            risk_col: Column name for risk scores
            
        Returns:
            NetworkX graph
        """
        self.graph = nx.Graph()
        
        # Add nodes with risk as weight
        for _, zone in zone_data.iterrows():
            h3_idx = zone.get('h3_index')
            if h3_idx is None:
                continue
                
            risk = zone.get(risk_col, zone.get('base_risk_score', 0.5))
            lat = zone.get('center_lat')
            lon = zone.get('center_lon')
            
            self.graph.add_node(h3_idx, risk=risk, lat=lat, lon=lon)
            self.zone_risks[h3_idx] = risk
        
        # Add edges between neighbors
        for h3_idx in self.graph.nodes():
            neighbors = self.h3_processor.get_neighbors(h3_idx, ring_size=1)
            for neighbor in neighbors:
                if neighbor in self.graph.nodes() and neighbor != h3_idx:
                    # Edge weight based on average risk of connected nodes
                    risk1 = self.zone_risks.get(h3_idx, 0.5)
                    risk2 = self.zone_risks.get(neighbor, 0.5)
                    weight = 1 + (risk1 + risk2)  # Higher risk = higher cost
                    self.graph.add_edge(h3_idx, neighbor, weight=weight)
        
        return self.graph
    
    def find_optimal_route(self, start_h3, end_h3, avoid_high_risk=True):
        """
        Find optimal route between two zones
        
        Args:
            start_h3: Starting H3 index
            end_h3: Destination H3 index
            avoid_high_risk: Whether to avoid high-risk zones
            
        Returns:
            Dictionary with route information
        """
        if self.graph is None:
            raise ValueError("Routing graph not built. Call build_routing_graph first.")
        
        if start_h3 not in self.graph or end_h3 not in self.graph:
            return {'status': 'error', 'message': 'Start or end zone not in graph'}
        
        try:
            if avoid_high_risk:
                # Use weighted shortest path (considers risk)
                path = nx.dijkstra_path(self.graph, start_h3, end_h3, weight='weight')
                path_length = nx.dijkstra_path_length(self.graph, start_h3, end_h3, weight='weight')
            else:
                # Simple shortest path
                path = nx.shortest_path(self.graph, start_h3, end_h3)
                path_length = len(path)
            
            # Calculate route metrics
            route_risks = [self.zone_risks.get(h3, 0) for h3 in path]
            
            route_info = {
                'status': 'success',
                'start': start_h3,
                'end': end_h3,
                'path': path,
                'path_length': len(path),
                'weighted_cost': path_length,
                'avg_risk': np.mean(route_risks),
                'max_risk': max(route_risks),
                'high_risk_zones': sum(1 for r in route_risks if r > 0.6),
                'coordinates': self._get_path_coordinates(path),
                'timestamp': datetime.now()
            }
            
            self.routing_history.append(route_info)
            return route_info
            
        except nx.NetworkXNoPath:
            return {'status': 'error', 'message': 'No path found between zones'}
    
    def find_alternative_routes(self, start_h3, end_h3, n_routes=3):
        """
        Find multiple alternative routes
        
        Args:
            start_h3: Starting H3 index
            end_h3: Destination H3 index
            n_routes: Number of alternative routes
            
        Returns:
            List of route dictionaries
        """
        routes = []
        
        # Primary route (risk-aware)
        primary = self.find_optimal_route(start_h3, end_h3, avoid_high_risk=True)
        if primary['status'] == 'success':
            primary['route_type'] = 'primary_safe'
            routes.append(primary)
        
        # Shortest route (may go through high risk)
        shortest = self.find_optimal_route(start_h3, end_h3, avoid_high_risk=False)
        if shortest['status'] == 'success' and shortest['path'] != primary.get('path'):
            shortest['route_type'] = 'shortest'
            routes.append(shortest)
        
        # Find additional routes by temporarily removing edges
        if len(routes) < n_routes and primary['status'] == 'success':
            temp_graph = self.graph.copy()
            
            for i in range(min(n_routes - len(routes), 2)):
                # Remove an edge from primary path
                if len(primary['path']) > 2:
                    idx = len(primary['path']) // 2
                    u, v = primary['path'][idx], primary['path'][idx + 1]
                    if temp_graph.has_edge(u, v):
                        temp_graph.remove_edge(u, v)
                        
                        try:
                            alt_path = nx.dijkstra_path(temp_graph, start_h3, end_h3, weight='weight')
                            if alt_path not in [r['path'] for r in routes]:
                                route_risks = [self.zone_risks.get(h3, 0) for h3 in alt_path]
                                routes.append({
                                    'status': 'success',
                                    'route_type': f'alternative_{i+1}',
                                    'path': alt_path,
                                    'path_length': len(alt_path),
                                    'avg_risk': np.mean(route_risks),
                                    'max_risk': max(route_risks),
                                    'coordinates': self._get_path_coordinates(alt_path)
                                })
                        except:
                            pass
        
        return routes
    
    def generate_routing_directive(self, incident_location, destination_type='hospital'):
        """
        Generate routing directive for emergency response
        
        Args:
            incident_location: H3 index or (lat, lon) of incident
            destination_type: Type of destination (hospital, station, etc.)
            
        Returns:
            Routing directive dictionary
        """
        # Convert coordinates to H3 if needed
        if isinstance(incident_location, tuple):
            start_h3 = self.h3_processor.lat_lon_to_h3(
                incident_location[0], incident_location[1]
            )
        else:
            start_h3 = incident_location
        
        # Find nearest destination of type
        # For now, use a sample destination (in real system, would query actual locations)
        destinations = self._get_destinations(destination_type)
        
        if not destinations:
            return {'status': 'error', 'message': f'No {destination_type} found'}
        
        # Find routes to all destinations
        best_route = None
        best_cost = float('inf')
        
        for dest in destinations:
            route = self.find_optimal_route(start_h3, dest['h3_index'])
            if route['status'] == 'success' and route['weighted_cost'] < best_cost:
                best_cost = route['weighted_cost']
                best_route = route
                best_route['destination_name'] = dest['name']
        
        if best_route is None:
            return {'status': 'error', 'message': 'No route found to any destination'}
        
        directive = {
            'incident_location': start_h3,
            'destination': best_route['end'],
            'destination_name': best_route.get('destination_name', 'Unknown'),
            'destination_type': destination_type,
            'recommended_route': best_route['path'],
            'estimated_zones': best_route['path_length'],
            'risk_assessment': {
                'avg_risk': best_route['avg_risk'],
                'max_risk': best_route['max_risk'],
                'high_risk_zones': best_route['high_risk_zones']
            },
            'coordinates': best_route['coordinates'],
            'alternatives': self.find_alternative_routes(start_h3, best_route['end']),
            'timestamp': datetime.now()
        }
        
        return directive
    
    def _get_path_coordinates(self, path):
        """Get coordinates for path visualization"""
        coords = []
        for h3_idx in path:
            if h3_idx in self.graph.nodes():
                node = self.graph.nodes[h3_idx]
                coords.append({
                    'h3_index': h3_idx,
                    'lat': node.get('lat'),
                    'lon': node.get('lon'),
                    'risk': self.zone_risks.get(h3_idx, 0)
                })
        return coords
    
    def _get_destinations(self, destination_type):
        """Get list of destinations by type"""
        # Sample destinations (in real system, would be from database)
        if destination_type == 'hospital':
            return [
                {'name': 'NYC Hospital A', 'h3_index': list(self.graph.nodes())[0] if self.graph.nodes() else None},
                {'name': 'NYC Hospital B', 'h3_index': list(self.graph.nodes())[-1] if self.graph.nodes() else None}
            ]
        return []
    
    def export_routing_directive(self, directive, filepath=None):
        """
        Export routing directive to CSV
        
        Args:
            directive: Routing directive dictionary
            filepath: Output path
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(OUTPUTS_DIR, f'priority_routing_directive_{timestamp}.csv')
        
        # Convert to DataFrame
        route_df = pd.DataFrame(directive['coordinates'])
        route_df['destination'] = directive['destination_name']
        route_df['destination_type'] = directive['destination_type']
        
        route_df.to_csv(filepath, index=False)
        print(f"Routing directive exported to {filepath}")
        return filepath
    
    def get_zone_accessibility(self, target_h3, max_distance=5):
        """
        Calculate accessibility score for a zone
        
        Args:
            target_h3: Target H3 index
            max_distance: Maximum path length to consider
            
        Returns:
            Accessibility metrics
        """
        if self.graph is None or target_h3 not in self.graph:
            return {}
        
        # Calculate shortest paths to all reachable nodes
        try:
            lengths = nx.single_source_dijkstra_path_length(
                self.graph, target_h3, cutoff=max_distance, weight='weight'
            )
        except:
            return {}
        
        reachable = len(lengths)
        avg_distance = np.mean(list(lengths.values())) if lengths else 0
        
        return {
            'zone': target_h3,
            'reachable_zones': reachable,
            'avg_weighted_distance': avg_distance,
            'accessibility_score': reachable / max(len(self.graph.nodes()), 1)
        }
