from flask import Flask, render_template, request, jsonify
import pandas as pd
import folium
import numpy as np
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler
import heapq

app = Flask(__name__)

# Load airport data
airports = pd.read_csv('airports.csv').dropna(subset=['name', 'iata_code', 'municipality', 'iso_country', 'coordinates'])
airports.sort_values(by='name', inplace=True)

# Extract latitude and longitude from 'coordinates' column
airports[['longitude_deg', 'latitude_deg']] = airports['coordinates'].str.split(',', expand=True).astype(float)

class RiskModel:
    def __init__(self, resolution=1.0):
        self.resolution = resolution
        self.lat_grid = np.arange(-90, 90, resolution)
        self.lon_grid = np.arange(-180, 180, resolution)
        self.risk_grid = np.ones((len(self.lat_grid), len(self.lon_grid)))
        
    def add_risks(self):
        birds = np.random.normal(0, 30, (1000, 2))
        tree = cKDTree(birds)
        for i, lat in enumerate(self.lat_grid):
            for j, lon in enumerate(self.lon_grid):
                self.risk_grid[i, j] += len(tree.query_ball_point([lat, lon], 500/111))
        self.risk_grid = MinMaxScaler(feature_range=(1,10)).fit_transform(self.risk_grid)

class PathFinder:
    def __init__(self, risk_model):
        self.risk_model = risk_model
        self.neighbors = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    
    def find_path(self, start, end):
        start_idx = self._coord_to_index(start)
        end_idx = self._coord_to_index(end)
        heap = []
        heapq.heappush(heap, (0, start_idx))
        came_from = {}
        cost_so_far = {start_idx: 0}
        
        while heap:
            current = heapq.heappop(heap)[1]
            if current == end_idx:
                return self._reconstruct_path(came_from, current)
            for neighbor in self._get_neighbors(current):
                new_cost = cost_so_far[current] + self.risk_model.risk_grid[neighbor]
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(heap, (new_cost + self._heuristic(end_idx, neighbor), neighbor))
                    came_from[neighbor] = current
        return None
    
    def _coord_to_index(self, coord):
        lat_idx = np.abs(self.risk_model.lat_grid - coord[0]).argmin()
        lon_idx = np.abs(self.risk_model.lon_grid - coord[1]).argmin()
        return (lat_idx, lon_idx)
    
    def _heuristic(self, a, b):
        return np.hypot(a[0]-b[0], a[1]-b[1])
    
    def _get_neighbors(self, node):
        return [(node[0]+dx, node[1]+dy) for dx, dy in self.neighbors 
                if 0 <= node[0]+dx < len(self.risk_model.lat_grid) and 
                0 <= node[1]+dy < len(self.risk_model.lon_grid)]
    
    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/planner')
def planner():
    return render_template('planner.html')

@app.route('/get-airports')
def get_airports():
    return jsonify(airports[['ident', 'name', 'iata_code']].to_dict('records'))

@app.route('/find-path', methods=['POST'])
def find_path():
    data = request.json
    try:
        start = airports[airports['ident'] == data['startId']].iloc[0]
        end = airports[airports['ident'] == data['endId']].iloc[0]
        start_coord = (start['latitude_deg'], start['longitude_deg'])
        end_coord = (end['latitude_deg'], end['longitude_deg'])
        
        risk_model = RiskModel(1.0)
        risk_model.add_risks()
        finder = PathFinder(risk_model)
        path = finder.find_path(start_coord, end_coord)
        
        m = folium.Map(location=start_coord, zoom_start=5)
        folium.Marker(start_coord, popup=f"Start: {start['name']}").add_to(m)
        folium.Marker(end_coord, popup=f"End: {end['name']}").add_to(m)
        
        if path:
            path_coords = [(risk_model.lat_grid[i], risk_model.lon_grid[j]) for i, j in path]
            folium.PolyLine(path_coords, color='blue', weight=2.5).add_to(m)
            m.fit_bounds([start_coord, end_coord])
        
        return jsonify({
            'map_html': m._repr_html_(),
            'start_name': start['name'],
            'end_name': end['name']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
