# world.py

from OpenGL.GL import *
import numpy as np
import numba
import pyrr
import ctypes

from settings import CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH, CUBE_VERTICES_DATA, CUBE_INDICES

# ... create_mesh_data fonksiyonu (değişiklik yok)
@numba.jit(nopython=True, cache=True)
def create_mesh_data(voxels):
    vertex_data_list = []
    for x in range(CHUNK_WIDTH):
        for y in range(CHUNK_HEIGHT):
            for z in range(CHUNK_DEPTH):
                if voxels[x, y, z] == 0:
                    continue
                # Ön Yüz
                if z == CHUNK_DEPTH - 1 or voxels[x, y, z + 1] == 0:
                    for i in range(0, 6):
                        idx = CUBE_INDICES[i]
                        vertex_data_list.extend([CUBE_VERTICES_DATA[idx, 0] + x, CUBE_VERTICES_DATA[idx, 1] + y, CUBE_VERTICES_DATA[idx, 2] + z, 0,0,1])
                # Arka Yüz
                if z == 0 or voxels[x, y, z - 1] == 0:
                    for i in range(6, 12):
                        idx = CUBE_INDICES[i]
                        vertex_data_list.extend([CUBE_VERTICES_DATA[idx, 0] + x, CUBE_VERTICES_DATA[idx, 1] + y, CUBE_VERTICES_DATA[idx, 2] + z, 0,0,-1])
                # Üst Yüz
                if y == CHUNK_HEIGHT - 1 or voxels[x, y + 1, z] == 0:
                    for i in range(12, 18):
                        idx = CUBE_INDICES[i]
                        vertex_data_list.extend([CUBE_VERTICES_DATA[idx, 0] + x, CUBE_VERTICES_DATA[idx, 1] + y, CUBE_VERTICES_DATA[idx, 2] + z, 0,1,0])
                # Alt Yüz
                if y == 0 or voxels[x, y - 1, z] == 0:
                    for i in range(18, 24):
                        idx = CUBE_INDICES[i]
                        vertex_data_list.extend([CUBE_VERTICES_DATA[idx, 0] + x, CUBE_VERTICES_DATA[idx, 1] + y, CUBE_VERTICES_DATA[idx, 2] + z, 0,-1,0])
                # Sağ Yüz
                if x == CHUNK_WIDTH - 1 or voxels[x + 1, y, z] == 0:
                    for i in range(24, 30):
                        idx = CUBE_INDICES[i]
                        vertex_data_list.extend([CUBE_VERTICES_DATA[idx, 0] + x, CUBE_VERTICES_DATA[idx, 1] + y, CUBE_VERTICES_DATA[idx, 2] + z, 1,0,0])
                # Sol Yüz
                if x == 0 or voxels[x - 1, y, z] == 0:
                    for i in range(30, 36):
                        idx = CUBE_INDICES[i]
                        vertex_data_list.extend([CUBE_VERTICES_DATA[idx, 0] + x, CUBE_VERTICES_DATA[idx, 1] + y, CUBE_VERTICES_DATA[idx, 2] + z, -1,0,0])
    return np.array(vertex_data_list, dtype=np.float32)

class Chunk:
    # ... __init__ ve build_mesh (değişiklik yok)
    def __init__(self, position):
        self.position = position
        self.voxels = np.zeros((CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH), dtype=np.uint8)
        self.is_dirty = True
        self.vao = None
        self.vbo = None
        self.vertex_count = 0
    def build_mesh(self):
        if not self.is_dirty: return
        mesh_data = create_mesh_data(self.voxels)
        self.vertex_count = len(mesh_data) // 6
        if self.vertex_count == 0:
            self.is_dirty = False
            return
        if self.vao is None: self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        if self.vbo is None: self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, mesh_data.nbytes, mesh_data, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        
        # <<< İYİ PRATİK DÜZELTMESİ: İş bittikten sonra buffer'ı serbest bırak >>>
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0) # VAO'yu da burada serbest bırakmak daha temizdir
        
        self.is_dirty = False
    def draw(self, shader):
        self.build_mesh()
        if self.vertex_count > 0:
            model_matrix = pyrr.matrix44.create_from_translation(self.position * CHUNK_WIDTH)
            glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model_matrix)
            glBindVertexArray(self.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
            # Bu satırın burada olması kritik önem taşıyor!
            glBindVertexArray(0)

# ... World sınıfının geri kalanı (değişiklik yok)
class World:
    def __init__(self):
        self.chunks = {}
        self.spawn_initial_chunks()
    def spawn_initial_chunks(self):
        for x in range(-2, 3):
            for z in range(-2, 3):
                pos = pyrr.Vector3([x, 0, z], dtype=np.int32)
                chunk = Chunk(pos)
                chunk.voxels[:, :CHUNK_HEIGHT // 2, :] = 1
                self.chunks[tuple(pos)] = chunk
    def get_voxel_from_world_pos(self, pos):
        try:
            chunk_pos = tuple(np.floor(pos / CHUNK_WIDTH).astype(int))
            local_pos = tuple((pos % CHUNK_WIDTH).astype(int))
            return self.chunks[chunk_pos].voxels[local_pos]
        except (KeyError, IndexError):
            return 0
    def set_voxel_from_world_pos(self, pos, block_type):
        try:
            chunk_pos = tuple(np.floor(pos / CHUNK_WIDTH).astype(int))
            local_pos = tuple((pos % CHUNK_WIDTH).astype(int))
            chunk = self.chunks[chunk_pos]
            if chunk.voxels[local_pos] == block_type: return
            chunk.voxels[local_pos] = block_type
            chunk.is_dirty = True
            lx, _, lz = local_pos
            if lx == 0: self.mark_chunk_dirty(chunk_pos[0] - 1, chunk_pos[1], chunk_pos[2])
            if lx == CHUNK_WIDTH - 1: self.mark_chunk_dirty(chunk_pos[0] + 1, chunk_pos[1], chunk_pos[2])
            if lz == 0: self.mark_chunk_dirty(chunk_pos[0], chunk_pos[1], chunk_pos[2] - 1)
            if lz == CHUNK_DEPTH - 1: self.mark_chunk_dirty(chunk_pos[0], chunk_pos[1], chunk_pos[2] + 1)
        except (KeyError, IndexError):
            pass
    def mark_chunk_dirty(self, cx, cy, cz):
        pos = (cx, cy, cz)
        if pos in self.chunks:
            self.chunks[pos].is_dirty = True
    def modify_voxels_in_radius(self, center_pos, brush_size, block_type):
        if brush_size == 1:
            self.set_voxel_from_world_pos(center_pos, block_type)
            return
        radius = (brush_size -1) // 2
        for x_offset in range(-radius, radius + 1):
            for y_offset in range(-radius, radius + 1):
                for z_offset in range(-radius, radius + 1):
                    target_pos = center_pos + np.array([x_offset, y_offset, z_offset])
                    self.set_voxel_from_world_pos(target_pos, block_type)
    def draw(self, shader):
        for chunk in self.chunks.values():
            chunk.draw(shader)
    def raycast(self, start, direction, max_dist=30):
        pos = np.floor(start).astype(int)
        step = np.sign(direction).astype(int)
        direction_safe = direction.copy()
        direction_safe[direction_safe == 0] = 1e-6
        t_delta = abs(1.0 / direction_safe)
        t_max = (np.sign(direction) * (np.floor(start) - start) + (np.sign(direction) * 0.5 + 0.5)) * t_delta
        prev_pos = None
        for _ in range(int(max_dist * 2)):
            if self.get_voxel_from_world_pos(pos) != 0:
                return tuple(pos), tuple(prev_pos) if prev_pos is not None else None
            prev_pos = pos.copy()
            if t_max[0] < t_max[1] and t_max[0] < t_max[2]:
                t_max[0] += t_delta[0]
                pos[0] += step[0]
            elif t_max[1] < t_max[2]:
                t_max[1] += t_delta[1]
                pos[1] += step[1]
            else:
                t_max[2] += t_delta[2]
                pos[2] += step[2]
        return None, None