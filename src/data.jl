# Data layer: real cities -> adjacency graph -> synthetic cities.

# TODO: load_geojson(path) -> City
# TODO: build_adjacency(polygons) -> Graph  (units sharing a border are neighbours)
# TODO: node_features(city) -> Matrix  (normalised; see spec for feature list)
# TODO: generate_synthetic_city(source_cities, n_units; rng) -> City
# TODO: save_city / load_city (JLD2 or JSON)
