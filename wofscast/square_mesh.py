"""Utils for creating triangular meshes for a limited area domain. """

import itertools
from typing import List, NamedTuple, Sequence, Tuple

import numpy as np
from scipy.spatial import transform
from scipy.spatial import Delaunay

import scipy.spatial
import trimesh

class TriangularMesh(NamedTuple):
  """Data structure for triangular meshes.

  Attributes:
    vertices: spatial positions of the vertices of the mesh of shape
        [num_vertices, num_dims].
    faces: triangular faces of the mesh of shape [num_faces, 3]. Contains
        integer indices into `vertices`.

  """
  vertices: np.ndarray
  faces: np.ndarray

from collections import defaultdict

def check_mesh_closed(faces: np.ndarray) -> bool:
    """Checks if the mesh is closed by ensuring every edge is shared by two faces."""
    edge_count = defaultdict(int)

    # Count occurrences of each edge
    for face in faces:
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        for edge in edges:
            # Sort the edge so (a, b) and (b, a) are treated the same
            edge = tuple(sorted(edge))
            edge_count[edge] += 1

    # Check if any edge is only used once (open edge)
    for edge, count in edge_count.items():
        if count != 2:
            print(f"Edge {edge} is only used {count} times, indicating an open mesh.")
            return False
    
    print("The mesh is closed.")
    return True

    
    
def get_meshes(start_lat, start_lon, domain_size, mesh_size=5):
    # Generate mesh hierarchy for the given patch
    meshes = get_hierarchy_of_triangular_meshes(
        splits=mesh_size, domain_size=domain_size, tiling=None)
    
    # Adjust mesh vertices to start from specific coordinates
    for mesh in meshes:
        mesh.vertices[:, 0] += start_lat
        mesh.vertices[:, 1] += start_lon
    
    return meshes

def concatenate_mesh_levels(mesh_levels_list):
    """Combines multiple mesh levels into one set of hierarchical meshes."""
    num_levels = len(mesh_levels_list[0])
    combined_meshes = []

    for level in range(num_levels):
        combined_vertices = np.array([], dtype=np.float32).reshape(0, 2)
        combined_faces = np.array([], dtype=np.int32).reshape(0, 3)
        vertex_offset = 0

        for mesh_levels in mesh_levels_list:
            mesh = mesh_levels[level]
            adjusted_faces = mesh.faces + vertex_offset
            combined_faces = np.vstack([combined_faces, adjusted_faces])
            combined_vertices = np.vstack([combined_vertices, mesh.vertices])
            vertex_offset += mesh.vertices.shape[0]

        combined_mesh = TriangularMesh(vertices=combined_vertices, faces=combined_faces)
        combined_meshes.append(combined_mesh)
    
    return combined_meshes

def get_hierarchy_of_tiled_triangular_meshes(tiling, domain_size, mesh_size=5):
    """Generates meshes for each patch and combines them into a hierarchy."""
    mesh_levels_list = []

    for i in range(tiling[0]):
        for j in range(tiling[1]):
            x_start = i * domain_size
            y_start = j * domain_size
            meshes = get_meshes(x_start, y_start, domain_size, mesh_size)
            mesh_levels_list.append(meshes)
    
    combined_meshes = concatenate_mesh_levels(mesh_levels_list)
    
    for i in range(len(combined_meshes)):
        combined_meshes[i] = add_boundary_triangles(combined_meshes[i], tiling, domain_size)

    return combined_meshes

def merge_tiled_meshes(mesh_list: Sequence[TriangularMesh]) -> TriangularMesh:
    """Merges all meshes into one. Assumes the last mesh is the finest.

    Args:
        mesh_list: Sequence of meshes, from coarse to fine refinement levels. The
        vertices and faces may contain those from preceding, coarser levels.

    Returns:
        `TriangularMesh` for which the vertices correspond to the highest
        resolution mesh in the hierarchy, and the faces are the join set of the
        faces at all levels of the hierarchy.
    """
    combined_vertices = np.array([], dtype=np.float32).reshape(0, 2)
    combined_faces = np.array([], dtype=np.int32).reshape(0, 3)
    vertex_offset = 0

    for mesh in mesh_list:
        adjusted_faces = mesh.faces + vertex_offset
        combined_faces = np.vstack([combined_faces, adjusted_faces])
        combined_vertices = np.vstack([combined_vertices, mesh.vertices])
        vertex_offset += mesh.vertices.shape[0]

    return TriangularMesh(vertices=combined_vertices, faces=combined_faces)

def add_boundary_triangles(mesh: TriangularMesh, tiling, domain_size, offset=2):
    """Add triangles along the boundaries of the tiled mesh patches, respecting the offset."""
    vertices = mesh.vertices
    faces = list(mesh.faces)
    
    # Define boundaries, adjusting for the offsets
    x_boundaries = [i * domain_size for i in range(1, tiling[0])]
    y_boundaries = [j * domain_size for j in range(1, tiling[1])]
    
    # Handle vertical boundaries (connecting patches left to right)
    for x_boundary in x_boundaries:
        left_boundary_vertices = np.where(vertices[:, 0] == x_boundary + offset)[0]
        right_boundary_vertices = np.where(vertices[:, 0] == x_boundary - offset)[0]  # Adjusting for offset
        
        # Sort vertices by their y-coordinate to match them correctly
        left_boundary_vertices = left_boundary_vertices[np.argsort(vertices[left_boundary_vertices, 1])]
        right_boundary_vertices = right_boundary_vertices[np.argsort(vertices[right_boundary_vertices, 1])]
        
        for idx in range(len(left_boundary_vertices)):
            p1 = left_boundary_vertices[idx]
            adj_p1 = right_boundary_vertices[idx]
            # Find next vertex along the boundary in the current patch
            if idx + 1 < len(left_boundary_vertices):
                p2 = left_boundary_vertices[idx + 1]
                adj_p2 = right_boundary_vertices[idx + 1]
                faces.append([p1, adj_p1, p2])
                faces.append([p2, adj_p1, adj_p2])
    
    # Handle horizontal boundaries (connecting patches top to bottom)
    for y_boundary in y_boundaries:
        bottom_boundary_vertices = np.where(vertices[:, 1] == y_boundary + offset)[0]
        top_boundary_vertices = np.where(vertices[:, 1] == y_boundary - offset)[0]  # Adjusting for offset
        
        # Sort vertices by their x-coordinate to match them correctly
        bottom_boundary_vertices = bottom_boundary_vertices[np.argsort(vertices[bottom_boundary_vertices, 0])]
        top_boundary_vertices = top_boundary_vertices[np.argsort(vertices[top_boundary_vertices, 0])]
        
        for idx in range(len(bottom_boundary_vertices)):
            p1 = bottom_boundary_vertices[idx]
            adj_p1 = top_boundary_vertices[idx]
            # Find next vertex along the boundary in the current patch
            if idx + 1 < len(bottom_boundary_vertices):
                p2 = bottom_boundary_vertices[idx + 1]
                adj_p2 = top_boundary_vertices[idx + 1]
                faces.append([p1, adj_p1, p2])
                faces.append([p2, adj_p1, adj_p2])
    
    return TriangularMesh(vertices=vertices, faces=np.array(faces))

def merge_meshes(
    mesh_list: Sequence[TriangularMesh]) -> TriangularMesh:
  """Merges all meshes into one. Assumes the last mesh is the finest.

  Args:
     mesh_list: Sequence of meshes, from coarse to fine refinement levels. The
       vertices and faces may contain those from preceding, coarser levels.

  Returns:
     `TriangularMesh` for which the vertices correspond to the highest
     resolution mesh in the hierarchy, and the faces are the join set of the
     faces at all levels of the hierarchy.
  """
  # MLF: Commented this out to test patching the mesh grid!
  #for mesh_i, mesh_ip1 in itertools.pairwise(mesh_list):
  #  num_nodes_mesh_i = mesh_i.vertices.shape[0]
  #  assert np.allclose(mesh_i.vertices, mesh_ip1.vertices[:num_nodes_mesh_i])

  return TriangularMesh(
      vertices=mesh_list[-1].vertices,
      faces=np.concatenate([mesh.faces for mesh in mesh_list], axis=0))


def concatenate_meshes(tiling: Tuple[int, int], domain_size: int, 
                       offset: int = 2, legacy=True
                      ) -> TriangularMesh:
    """
    Concatenate multiple triangular meshes to form a larger tiled mesh.
    
    Args:
        tiling: Number of tiles in the x and y directions.
        domain_size: Size of the domain for each tile.
        
    Returns:
        A concatenated TriangularMesh.
    """
    nx, ny = tiling
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    for i in range(nx):
        for j in range(ny):
            mesh = get_tri_mesh(i * domain_size, j * domain_size, domain_size, offset=offset, legacy=legacy)
            all_vertices.append(mesh.vertices)
            all_faces.append(mesh.faces + vertex_offset)
            vertex_offset += mesh.vertices.shape[0]
    
    concatenated_vertices = np.vstack(all_vertices)
    concatenated_faces = np.vstack(all_faces)
    
    return TriangularMesh(vertices=concatenated_vertices, faces=concatenated_faces)

def get_hierarchy_of_triangular_meshes(
    splits: int, 
    domain_size: int, 
    tiling=None, 
    legacy_mesh: bool = False
) -> List[TriangularMesh]:
    """Returns a sequence of meshes

      Starting with a regular icosahedron (12 vertices, 20 faces, 30 edges) with
      circumscribed unit sphere. Then, each triangular face is iteratively
      subdivided into 4 triangular faces `splits` times. The new vertices are then
      projected back onto the unit sphere. All resulting meshes are returned in a
      list, from lowest to highest resolution.

      The vertices in each face are specified in counter-clockwise order as
      observed from the outside the icosahedron.

      Args:
         splits: How many times to split each triangle.
         domain_size : int: Number of grid points 
         tiling : 2-tuple of int: Whether to tile the initial mesh (default=None). Used for applying 
             the trained limited area model over a larger domain. 
      Returns:
         Sequence of `TriangularMesh`s of length `splits + 1` each with:

           vertices: [num_vertices, 3] vertex positions in 3D, all with unit norm.
           faces: [num_faces, 3] with triangular faces joining sets of 3 vertices.
           Each row contains three indices into the vertices array, indicating
           the vertices adjacent to the face. Always with positive orientation
           (counterclock-wise when looking from the outside).
    """
    if tiling:
        current_mesh = concatenate_meshes(tiling, domain_size, legacy=legacy_mesh)
    else:
        current_mesh = get_tri_mesh(0, 0, domain_size, offset=2, legacy=legacy_mesh)
        # The faces are not closed, so had to add additional edges 
        # manually in the face_to_edges function below.
        #is_closed = check_mesh_closed(current_mesh.faces)
        
    output_meshes = [current_mesh]
    
    for _ in range(splits):
        current_mesh = _two_split_triangle_faces(current_mesh)
        output_meshes.append(current_mesh)
    
    return output_meshes

def enforce_consistent_orientation(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Ensures all triangular faces have a counterclockwise orientation."""
    for i, face in enumerate(faces):
        # Extract the vertices of the current face
        v0, v1, v2 = vertices[face]

        # Compute the cross product of the two edge vectors
        cross_product = np.cross(v1 - v0, v2 - v0)

        # If the cross product is negative, reverse the order to make it counterclockwise
        if cross_product < 0:
            faces[i] = [face[0], face[2], face[1]]  # Swap the second and third vertices
    return faces


def get_tri_mesh(x_start, y_start, size, offset=0, legacy: bool = False) -> TriangularMesh:
    """Returns a staggered triangular mesh.
  
    Returns:
        TriangularMesh with:

        vertices: [num_vertices=12, 3] vertex positions in 3D, all with unit norm.
        faces: [num_faces=20, 3] with triangular faces joining sets of 3 vertices.
         Each row contains three indices into the vertices array, indicating
         the vertices adjacent to the face. Always with positive orientation (
         counterclock-wise when looking from the outside).

    """    
    half_size = size // 2 
    
    vertices = np.array([
        [x_start+offset, y_start+offset], # Bottom left corner
        [x_start + size - offset, y_start+offset], # Bottom right corner
        [x_start + size - offset, y_start + size -offset], # Top right corner
        [x_start + offset, y_start + size - offset], # Top left corner 
        [x_start + half_size + offset, y_start + half_size + offset]  # center point
    ], dtype=np.float32)
    
    tri = Delaunay(vertices)
    faces = tri.simplices  # The faces are defined by the Delaunay triangulation
    
    # Enforce consistent counterclockwise orientation
    if not legacy:
        print('Enforcing consistent face orientation')
        faces = enforce_consistent_orientation(vertices, faces)
    
    return TriangularMesh(vertices=vertices,
                        faces=np.array(faces, dtype=np.int32))

def _two_split_triangle_faces(triangular_mesh: TriangularMesh) -> TriangularMesh:
    """Splits each triangular face into 4 triangles keeping the orientation."""
    new_vertices_builder = _ChildVerticesBuilder(triangular_mesh.vertices)

    new_faces = []
    for ind1, ind2, ind3 in triangular_mesh.faces:
        ind12 = new_vertices_builder.get_new_child_vertex_index((ind1, ind2))
        ind23 = new_vertices_builder.get_new_child_vertex_index((ind2, ind3))
        ind31 = new_vertices_builder.get_new_child_vertex_index((ind3, ind1))

        new_faces.extend([
            [ind1, ind12, ind31],  # 1
            [ind12, ind2, ind23],  # 2
            [ind31, ind23, ind3],  # 3
            [ind12, ind23, ind31],  # 4
        ])

    return TriangularMesh(vertices=new_vertices_builder.get_all_vertices(), 
                          faces=np.array(new_faces, dtype=np.int32))



class _ChildVerticesBuilder:
    """Bookkeeping of new child vertices added to an existing set of vertices for a 2D domain."""

    def __init__(self, parent_vertices):
        self._child_vertices_index_mapping = {}
        self._parent_vertices = parent_vertices
        self._all_vertices_list = list(parent_vertices)

    def _get_child_vertex_key(self, parent_vertex_indices):
        return tuple(sorted(parent_vertex_indices))

    def _create_child_vertex(self, parent_vertex_indices):
        """Creates a new vertex as the midpoint of the edge in 2D."""
        child_vertex_position = self._parent_vertices[list(parent_vertex_indices)].mean(axis=0)
        child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
        self._child_vertices_index_mapping[child_vertex_key] = len(self._all_vertices_list)
        self._all_vertices_list.append(child_vertex_position)

    def get_new_child_vertex_index(self, parent_vertex_indices):
        child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
        if child_vertex_key not in self._child_vertices_index_mapping:
            self._create_child_vertex(parent_vertex_indices)
        return self._child_vertices_index_mapping[child_vertex_key]

    def get_all_vertices(self):
        return np.array(self._all_vertices_list)


def make_edges_bi_directional(senders, receivers):
    # Create a set of (sender, receiver) pairs
    edge_pairs = set(zip(senders, receivers))

    # New lists to hold bi-directional edges
    new_senders = list(senders)
    new_receivers = list(receivers)

    # Check if all edges are bi-directional
    for sender, receiver in zip(senders, receivers):
        if (receiver, sender) not in edge_pairs:
            # Add the missing reverse edge
            new_senders.append(receiver)
            new_receivers.append(sender)
            #print(f"Added reverse edge for ({receiver}, {sender})")

    # Convert lists back to arrays
    return np.array(new_senders), np.array(new_receivers)
    
    
def faces_to_edges(faces: np.ndarray, legacy: bool = False) -> Tuple[np.ndarray, np.ndarray]:
  """Transforms polygonal faces to sender and receiver indices.

  It does so by transforming every face into N_i edges. Such if the triangular
  face has indices [0, 1, 2], three edges are added 0->1, 1->2, and 2->0.

  If all faces have consistent orientation, and the surface represented by the
  faces is closed, then every edge in a polygon with a certain orientation
  is also part of another polygon with the opposite orientation. In this
  situation, the edges returned by the method are always bidirectional.

  Args:
    faces: Integer array of shape [num_faces, 3]. Contains node indices
        adjacent to each face.
  Returns:
    Tuple with sender/receiver indices, each of shape [num_edges=num_faces*3].

  """
  assert faces.ndim == 2
  assert faces.shape[-1] == 3
  senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
  receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])

  # The triangular mesh faces have consistent orientations, but 
  # open faces prevent fully bi-directional edges. As a fix
  # we can add the reverse of an existing edge direction if it 
  # does not exist. As of yet to determine what some triangle faces
  # are not fully closed. 
  if not legacy:
      print('Correcting the bi-direction mesh edges') 
      senders, receivers = make_edges_bi_directional(senders, receivers)

  return senders, receivers


def get_mesh_coords(mesh, grid_lat, grid_lon):
    """Get linear interpolated latitude and longitude coordiantes of the 
       MESH nodes. 
    """
    vertices = mesh.vertices
    
    # Assuming vertices is a 2D numpy array where each row is a vertex and
    # columns correspond to the x and y coordinates (in the 0-500 range)
    normalized_vertices = vertices / np.max(vertices)  # Normalize to 0-1 range

    # Linearly interpolate to get the latitude and longitude
    # lat_hr and lon_hr are high-resolution latitude and longitude arrays
    mesh_nodes_lat = grid_lat.min() + (grid_lat.max() - grid_lat.min()) * normalized_vertices[:, 0]
    mesh_nodes_lon = grid_lon.min() + (grid_lon.max() - grid_lon.min()) * normalized_vertices[:, 1]

    return mesh_nodes_lon, mesh_nodes_lat

def get_grid_positions(grid_size: int, add_3d_dim=False):
    """Generate grid positions for a given range in a 2D space."""
    x, y = np.meshgrid(range(grid_size), range(grid_size))
    if add_3d_dim:
        grid_positions = np.vstack([x.ravel(), y.ravel(), np.zeros(x.size)]).T
    else:
        grid_positions = np.vstack([x.ravel(), y.ravel()]).T
    return grid_positions
    

def radius_query_indices(
    grid_size : int, 
    finest_mesh : TriangularMesh, 
    radius: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Find mesh-grid edge indices within a given radius.

    Parameters:
    grid_size : int : Number of NWP grid positions in one direction
    finest_mesh : TriangularMesh : Finest mesh refinement layer in the multi-mesh
    radius : float : search radius in integer Caresian space
    
    Returns:
    A tuple of arrays containing the grid indices and mesh indices that are connected 
    together within the search radius. 
    """
    grid_positions = get_grid_positions(grid_size)
    mesh_positions = finest_mesh.vertices
    
    # Build a k-d tree for the mesh positions
    kd_tree = scipy.spatial.cKDTree(mesh_positions)

    # Query the k-d tree for all grid points within the radius of mesh points
    query_indices = kd_tree.query_ball_point(x=grid_positions, r=radius)

    # Generate the edge indices
    grid_edge_indices = []
    mesh_edge_indices = []
    for grid_index, mesh_neighbors in enumerate(query_indices):
        grid_edge_indices.append(np.repeat(grid_index, len(mesh_neighbors)))
        mesh_edge_indices.append(mesh_neighbors)

    # Convert to numpy arrays and flatten
    grid_edge_indices = np.concatenate(grid_edge_indices, axis=0).astype(int)
    mesh_edge_indices = np.concatenate(mesh_edge_indices, axis=0).astype(int)

    return grid_edge_indices, mesh_edge_indices

def in_mesh_triangle_indices(
    *,
    grid_size: int, 
    mesh: TriangularMesh) -> tuple[np.ndarray, np.ndarray]:
  """Returns mesh-grid edge indices for grid points contained in mesh triangles.

  Args:
    grid_latitude: Latitude values for the grid [num_lat_points]
    grid_longitude: Longitude values for the grid [num_lon_points]
    mesh: Mesh object.

  Returns:
    tuple with `grid_indices` and `mesh_indices` indicating edges between the
    grid and the mesh vertices of the triangle that contain each grid point.
    The number of edges is always num_lat_points * num_lon_points * 3
    * grid_indices: Indices of shape [num_edges], that index into a
      [num_lat_points, num_lon_points] grid, after flattening the leading axes.
    * mesh_indices: Indices of shape [num_edges], that index into mesh.vertices.
  """
  # Trimesh works in 3D. so need to add false 3rd dim for the code
  grid_positions = get_grid_positions(grid_size, add_3d_dim=True) 
  vertices = mesh.vertices
  
  x = vertices[:,0]
  y = vertices[:,1]

  vertices_3D = np.vstack([x.ravel(), y.ravel(), np.zeros(x.size)]).T

  mesh_trimesh = trimesh.Trimesh(vertices=vertices_3D, faces=mesh.faces)

  # [num_grid_points] with mesh face indices for each grid point.
  _, _, query_face_indices = trimesh.proximity.closest_point(
      mesh_trimesh, grid_positions)

  # [num_grid_points, 3] with mesh node indices for each grid point.
  mesh_edge_indices = mesh.faces[query_face_indices]

  # [num_grid_points, 3] with grid node indices, where every row simply contains
  # the row (grid_point) index.
  grid_indices = np.arange(grid_positions.shape[0])
  grid_edge_indices = np.tile(grid_indices.reshape([-1, 1]), [1, 3])

  # Flatten to get a regular list.
  # [num_edges=num_grid_points*3]
  mesh_edge_indices = mesh_edge_indices.reshape([-1])
  grid_edge_indices = grid_edge_indices.reshape([-1])

  return grid_edge_indices, mesh_edge_indices

