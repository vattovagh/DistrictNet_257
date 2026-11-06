# Core types shared by every other file.

# TODO: BasicUnit  -- one geographic unit (id, centroid, population/demand, area, ...)
# TODO: City       -- units + adjacency Graph + metadata (name, depot location)
# TODO: District   -- set of unit ids, must be connected and within capacity
# TODO: Partition  -- vector of Districts covering all units exactly once
# TODO: validate(partition, city) -> Bool  (contiguity, capacity, coverage)
