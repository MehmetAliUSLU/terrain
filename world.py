# world.py

from OpenGL.GL import *
import numpy as np
import numba
import pyrr
import ctypes
import noise
import os

from settings import (
    CHUNK_WIDTH,
    CHUNK_DEPTH,
    RENDER_DISTANCE,
    UNLOAD_DISTANCE,
    TERRAIN_SCALE,
    NOISE_OCTAVES,
    NOISE_PERSISTENCE,
    NOISE_LACUNARITY,
    HEIGHT_SCALE,
)
from history import EditAction

@numba.jit(nopython=True, cache=True)
def create_heightmap_mesh(heightmap):
    # --- 1. ADIM: Her bir verteks için doğru yönde normalleri biriktirmek ---
    normals = np.zeros((CHUNK_WIDTH + 1, CHUNK_DEPTH + 1, 3), dtype=np.float32)

    for x in range(CHUNK_WIDTH):
        for z in range(CHUNK_DEPTH):
            p1 = np.array([x,     heightmap[x, z],     z],     dtype=np.float32)
            p2 = np.array([x + 1, heightmap[x + 1, z], z],     dtype=np.float32)
            p3 = np.array([x,     heightmap[x, z + 1], z + 1], dtype=np.float32)
            p4 = np.array([x + 1, heightmap[x + 1, z + 1], z + 1], dtype=np.float32)

            # İlk üçgen (p1, p3, p2) - Saatın Tersi Yönünde (CCW)
            # Bu sıralama, yukarı (+Y) yönünde bir normal vektörü üretir.
            vec1 = p3 - p1
            vec2 = p2 - p1
            normal1 = np.cross(vec1, vec2)
            
            normals[x, z] += normal1
            normals[x, z + 1] += normal1
            normals[x + 1, z] += normal1

            # İkinci üçgen (p3, p4, p2) - Saatın Tersi Yönünde (CCW)
            vec3 = p4 - p3
            vec4 = p2 - p3
            normal2 = np.cross(vec3, vec4)

            normals[x, z + 1] += normal2
            normals[x + 1, z + 1] += normal2
            normals[x + 1, z] += normal2

    # --- 2. ADIM: Biriktirilen tüm normalleri normalize etmek ---
    for i in range(normals.shape[0]):
        for j in range(normals.shape[1]):
            norm = normals[i, j]
            mag = np.sqrt(norm[0]**2 + norm[1]**2 + norm[2]**2)
            if mag > 0:
                normals[i, j] = norm / mag

    # --- 3. ADIM: Nihai verteks verisini doğru sarım sırasıyla oluşturmak ---
    vertices = []
    dtype = np.float32
    for x in range(CHUNK_WIDTH):
        for z in range(CHUNK_DEPTH):
            pos1 = np.array([x,     heightmap[x, z],     z],     dtype=dtype)
            pos2 = np.array([x + 1, heightmap[x + 1, z], z],     dtype=dtype)
            pos3 = np.array([x,     heightmap[x, z + 1], z + 1], dtype=dtype)
            pos4 = np.array([x + 1, heightmap[x + 1, z + 1], z + 1], dtype=dtype)

            n1 = normals[x, z]
            n2 = normals[x + 1, z]
            n3 = normals[x, z + 1]
            n4 = normals[x + 1, z + 1]

            uv1 = np.array([x, z], dtype=dtype)
            uv2 = np.array([x + 1, z], dtype=dtype)
            uv3 = np.array([x, z + 1], dtype=dtype)
            uv4 = np.array([x + 1, z + 1], dtype=dtype)
            
            # 1. Üçgen (pos1, pos3, pos2) - Sarım sırası, normal hesaplamasıyla EŞLEŞMELİ
            vertices.extend([pos1[0], pos1[1], pos1[2], n1[0], n1[1], n1[2], uv1[0], uv1[1]])
            vertices.extend([pos3[0], pos3[1], pos3[2], n3[0], n3[1], n3[2], uv3[0], uv3[1]])
            vertices.extend([pos2[0], pos2[1], pos2[2], n2[0], n2[1], n2[2], uv2[0], uv2[1]])
            
            # 2. Üçgen (pos3, pos4, pos2) - Sarım sırası, normal hesaplamasıyla EŞLEŞMELİ
            vertices.extend([pos3[0], pos3[1], pos3[2], n3[0], n3[1], n3[2], uv3[0], uv3[1]])
            vertices.extend([pos4[0], pos4[1], pos4[2], n4[0], n4[1], n4[2], uv4[0], uv4[1]])
            vertices.extend([pos2[0], pos2[1], pos2[2], n2[0], n2[1], n2[2], uv2[0], uv2[1]])
            
    return np.array(vertices, dtype=dtype)

# --- Procedural terrain helpers ---
def procedural_height(world_x, world_z):
    n = noise.pnoise2(
        world_x / TERRAIN_SCALE,
        world_z / TERRAIN_SCALE,
        octaves=NOISE_OCTAVES,
        persistence=NOISE_PERSISTENCE,
        lacunarity=NOISE_LACUNARITY,
        base=0,
    )
    return (n + 1.0) * 0.5 * HEIGHT_SCALE

def bake_splatmap_for_chunk(chunk):
    for x in range(CHUNK_WIDTH + 1):
        for z in range(CHUNK_DEPTH + 1):
            h_dx = chunk.heightmap[min(x + 1, CHUNK_WIDTH), z] - chunk.heightmap[max(x - 1, 0), z]
            h_dz = chunk.heightmap[x, min(z + 1, CHUNK_DEPTH)] - chunk.heightmap[x, max(z - 1, 0)]
            normal_approx = np.array([-h_dx, 2.0, -h_dz], dtype=np.float32)
            norm = np.linalg.norm(normal_approx)
            if norm > 0:
                normal_approx /= norm
            slope = 1.0 - normal_approx[1]

            height = chunk.heightmap[x, z]

            slope_grass = 0.35
            slope_rock = 0.7
            height_dirt = 2.5

            dirt_blend_slope = np.clip((slope - (slope_grass - 0.1)) / 0.2, 0, 1)
            dirt_blend_height = np.clip((height - (height_dirt - 1.0)) / 2.0, 0, 1)
            dirt_blend = max(dirt_blend_slope, dirt_blend_height)

            grass_weight = 1.0 - dirt_blend
            dirt_weight = dirt_blend
            rock_blend = np.clip((slope - (slope_rock - 0.2)) / 0.4, 0, 1)

            final_grass = grass_weight * (1.0 - rock_blend)
            final_dirt = dirt_weight * (1.0 - rock_blend)
            final_rock = rock_blend

            chunk.splatmap[z, x] = [final_grass, final_dirt, final_rock]

class Chunk:
    def __init__(self, position):
        self.position = position
        self.heightmap = np.zeros((CHUNK_WIDTH + 1, CHUNK_DEPTH + 1), dtype=np.float32)
        self.splatmap = np.zeros(( CHUNK_DEPTH + 1,CHUNK_WIDTH + 1, 3), dtype=np.float32)
        self.splatmap_texture = None
        self.splatmap_is_dirty = True # Başlangıçta GPU'ya yüklenmesi gerek
        
        self.vao = None
        self.vbo = None
        self.vertex_count = 0
        self.is_dirty = True
        self.is_meshing = False

    def upload_mesh_to_gpu(self, mesh_data):
        # Bir vertex artık 8 float (32 byte)
        self.vertex_count = len(mesh_data) // 8
        if self.vertex_count == 0: return

        if self.vao is None: self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        if self.vbo is None: self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, mesh_data.nbytes, mesh_data, GL_DYNAMIC_DRAW)
        
        stride = 32 # (3+3+2) * 4 byte
        # Konum (Location 0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # Normal (Location 1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        # Doku Koordinatları (Location 2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def destroy(self):
        if self.vbo is not None:
            try:
                glDeleteBuffers(1, [self.vbo])
            except Exception:
                pass
            self.vbo = None
        if self.vao is not None:
            try:
                glDeleteVertexArrays(1, [self.vao])
            except Exception:
                pass
            self.vao = None
        if self.splatmap_texture is not None:
            try:
                glDeleteTextures(1, [self.splatmap_texture])
            except Exception:
                pass
            self.splatmap_texture = None

    def upload_splatmap_to_gpu(self):
        if self.splatmap_texture is None:
            self.splatmap_texture = glGenTextures(1)
        
        glActiveTexture(GL_TEXTURE3) # Başka bir doku birimi kullanalım (0,1,2 dolu)
        glBindTexture(GL_TEXTURE_2D, self.splatmap_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # NumPy dizisini GPU'ya yükle
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, CHUNK_WIDTH + 1, CHUNK_DEPTH + 1, 0, GL_RGB, GL_FLOAT, self.splatmap)
        self.splatmap_is_dirty = False

    def draw(self, shader):
        # Splatmap kirliyse (boyandıysa) GPU'yu güncelle
        if self.splatmap_is_dirty:
            self.upload_splatmap_to_gpu()
            
        if self.vertex_count > 0 and self.vao is not None:
            model_matrix = pyrr.matrix44.create_from_translation(
                pyrr.Vector3([self.position.x * CHUNK_WIDTH, 0, self.position.z * CHUNK_DEPTH])
            )
            glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model_matrix)
            
            # Splatmap dokusunu shader'a bağla
            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, self.splatmap_texture)
            
            glBindVertexArray(self.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
            glBindVertexArray(0)

class World:
    def __init__(self, request_queue):
        self.chunks = {}
        self.request_queue = request_queue
        # Persist edits across unload/load
        self.save_dir = os.path.join("saves", "chunks")
        os.makedirs(self.save_dir, exist_ok=True)
        self.spawn_initial_chunks()
        
        
    def get_normal_at_world_pos(self, world_x, world_z):
        """Verilen dünya koordinatındaki yüzey normalini yaklaşık olarak hesaplar."""
        chunk = self.get_chunk_at_world_pos(world_x, world_z)
        if not chunk:
            normal = pyrr.Vector3([h_l - h_r, 2.0, h_d - h_u])
            return pyrr.vector.normalize(normal)

        # Komşu piksellerin yüksekliklerini alarak eğimi hesapla
        h_l = self.get_height_at_world_pos(world_x - 1, world_z) # Sol
        h_r = self.get_height_at_world_pos(world_x + 1, world_z) # Sağ
        h_d = self.get_height_at_world_pos(world_x, world_z - 1) # Aşağı
        h_u = self.get_height_at_world_pos(world_x, world_z + 1) # Yukarı

        # Yüzey normalini dikey vektörlerin çapraz çarpımından türet
        normal = pyrr.Vector3([h_l - h_r, 2.0, h_d - h_u])
        return pyrr.vector.normalize(normal)
    
    def get_chunk_at_world_pos(self, world_x, world_z):
        """Verilen dünya koordinatındaki chunk'ı döndürür."""
        chunk_pos_x = int(np.floor(world_x / CHUNK_WIDTH))
        chunk_pos_z = int(np.floor(world_z / CHUNK_DEPTH))
        return self.chunks.get((chunk_pos_x, 0, chunk_pos_z))

    def spawn_initial_chunks(self):
        chunk_pos = (0, 0, 0)
        pos = pyrr.Vector3(chunk_pos, dtype=np.int32)
        chunk = Chunk(pos)
        
        # Yükseklik haritasını oluştur (bu kısım aynı)
        scale = 100.0; octaves = 6; persistence = 0.5; lacunarity = 2.0
        for x in range(CHUNK_WIDTH + 1):
            for z in range(CHUNK_DEPTH + 1):
                world_x = pos.x * CHUNK_WIDTH + x; world_z = pos.z * CHUNK_DEPTH + z
                noise_val = noise.pnoise2(world_x/scale, world_z/scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=0)
                chunk.heightmap[x, z] = (noise_val + 1) / 2 * 6 # 6 olarak güncelledik

        # Başlangıç splatmap'ini otomatik olarak oluştur
        for x in range(CHUNK_WIDTH + 1):
            for z in range(CHUNK_DEPTH + 1):
                # Komşu piksellerden eğimi hesapla
                h_dx = chunk.heightmap[min(x + 1, CHUNK_WIDTH), z] - chunk.heightmap[max(x - 1, 0), z]
                h_dz = chunk.heightmap[x, min(z + 1, CHUNK_DEPTH)] - chunk.heightmap[x, max(z - 1, 0)]
                normal_approx = np.array([-h_dx, 2.0, -h_dz]) # <--- This creates a numpy.ndarray
                normal_approx /= np.linalg.norm(normal_approx)
                slope = 1.0 - normal_approx[1] # <--- Indexing with [1] is correct for numpy

                height = chunk.heightmap[x, z]

                # Shader'daki kuralları burada Python'da taklit ediyoruz
                slope_grass = 0.35; slope_rock = 0.7; height_dirt = 2.5
                
                dirt_blend_slope = np.clip((slope - (slope_grass - 0.1)) / 0.2, 0, 1)
                dirt_blend_height = np.clip((height - (height_dirt - 1.0)) / 2.0, 0, 1)
                dirt_blend = max(dirt_blend_slope, dirt_blend_height)
                
                grass_weight = 1.0 - dirt_blend
                dirt_weight = dirt_blend

                rock_blend = np.clip((slope - (slope_rock - 0.2)) / 0.4, 0, 1)
                
                # Önceki karışımı kaya ile karıştır
                final_grass = grass_weight * (1.0 - rock_blend)
                final_dirt = dirt_weight * (1.0 - rock_blend)
                final_rock = rock_blend
                
                chunk.splatmap[z,x] = [final_grass, final_dirt, final_rock]

        self.chunks[chunk_pos] = chunk
        self.request_queue.put(chunk)

    def get_height_at_world_pos(self, world_x, world_z):
        try:
            chunk_pos_x = int(np.floor(world_x / CHUNK_WIDTH)); chunk_pos_z = int(np.floor(world_z / CHUNK_DEPTH))
            chunk = self.chunks.get((chunk_pos_x, 0, chunk_pos_z))
            if chunk is None:
                return procedural_height(world_x, world_z)
            local_x = world_x - chunk_pos_x * CHUNK_WIDTH; local_z = world_z - chunk_pos_z * CHUNK_DEPTH
            grid_x = int(np.floor(local_x)); grid_z = int(np.floor(local_z))
            x_frac = local_x - grid_x; z_frac = local_z - grid_z
            h00 = chunk.heightmap[grid_x, grid_z]; h10 = chunk.heightmap[grid_x + 1, grid_z]
            h01 = chunk.heightmap[grid_x, grid_z + 1]; h11 = chunk.heightmap[grid_x + 1, grid_z + 1]
            if x_frac + z_frac < 1:
                return h00 + (h10 - h00) * x_frac + (h01 - h00) * z_frac
            else:
                return h11 + (h01 - h11) * (1 - x_frac) + (h10 - h11) * (1 - z_frac)
        except (KeyError, IndexError):
            return procedural_height(world_x, world_z)


    def carve_river_along_path(self, path_points, settings, action):
        """
        Verilen bir yol boyunca dere yatağı oyar. Bu karmaşık ve geri alınabilir
        tek bir eylem olarak kaydedilir.
        """
        if len(path_points) < 2: return

        width = settings["width"]
        depth = settings["depth"]
        smoothing = settings["smoothing"]
        
        # 1. Yol boyunca enterpolasyon yaparak daha sık noktalar oluştur
        total_length = 0
        for i in range(len(path_points) - 1):
            total_length += np.linalg.norm(path_points[i+1] - path_points[i])
        
        interpolated_points = []
        num_steps = max(1, int(total_length)) # Her bir dünya biriminde bir nokta
        for i in range(num_steps + 1):
            progress = i / num_steps
            
            # Hangi segmentte olduğumuzu bul
            current_len = 0
            for j in range(len(path_points) - 1):
                segment_len = np.linalg.norm(path_points[j+1] - path_points[j])
                if current_len + segment_len >= progress * total_length:
                    # Bu segmentteyiz
                    segment_progress = (progress * total_length - current_len) / segment_len
                    p1 = path_points[j]
                    p2 = path_points[j+1]
                    point = p1 + (p2 - p1) * segment_progress
                    interpolated_points.append(point)
                    break
                current_len += segment_len

        if not interpolated_points: return

        # 2. Her bir enterpolasyonlu nokta etrafındaki araziyi değiştir
        points_to_modify = {}
        for point in interpolated_points:
            radius = width / 2.0
            for x in range(int(np.floor(point.x - radius)), int(np.ceil(point.x + radius))):
                for z in range(int(np.floor(point.z - radius)), int(np.ceil(point.z + radius))):
                    dist_sq = (x - point.x)**2 + (z - point.z)**2
                    if dist_sq > radius**2: continue
                    
                    # Hedef yükseklik: yolun o anki yüksekliğinden daha derin
                    target_height = point.y - depth

                    key = (int(round(x)), int(round(z)))
                    if key not in points_to_modify or target_height < points_to_modify[key]:
                        points_to_modify[key] = target_height

        # 3. De�i�iklikleri ve yumu�atmay� uygula (�ok-chunk)
        affected_chunks = set()
        for (x, z), target_h in points_to_modify.items():
            cx = int(np.floor(x / CHUNK_WIDTH))
            cz = int(np.floor(z / CHUNK_DEPTH))
            chunk = self.chunks.get((cx, 0, cz))
            if chunk is None:
                # Gerekirse chunk'� olu�tur (prosed�rel)
                pos = pyrr.Vector3([cx, 0, cz], dtype=np.int32)
                chunk = Chunk(pos)
                for xx in range(CHUNK_WIDTH + 1):
                    for zz in range(CHUNK_DEPTH + 1):
                        wx = cx * CHUNK_WIDTH + xx
                        wz = cz * CHUNK_DEPTH + zz
                        chunk.heightmap[xx, zz] = procedural_height(wx, wz)
                bake_splatmap_for_chunk(chunk)
                self.chunks[(cx, 0, cz)] = chunk
                chunk.is_meshing = True
                self.request_queue.put(chunk)

            local_x = int(round(x - chunk.position.x * CHUNK_WIDTH))
            local_z = int(round(z - chunk.position.z * CHUNK_DEPTH))

            if 0 <= local_x < CHUNK_WIDTH + 1 and 0 <= local_z < CHUNK_DEPTH + 1:
                old_h = chunk.heightmap[local_x, local_z]
                old_s = chunk.splatmap[local_z, local_x].copy()

                new_h = min(old_h, target_h)
                chunk.heightmap[local_x, local_z] = new_h

                target_s = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                mixed_s = old_s + (target_s - old_s) * 0.5
                chunk.splatmap[local_z, local_x] = mixed_s
                chunk.splatmap_is_dirty = True

                if local_x in (0, CHUNK_WIDTH) or local_z in (0, CHUNK_DEPTH):
                    self._mirror_border_update(chunk, local_x, local_z, new_h, mixed_s, action)

                action.record_change(chunk, local_x, local_z, old_h, new_h, old_s, mixed_s.copy())
                affected_chunks.add(chunk)

        # Etkilenen chunklar� mesh i�in kuyru�a al
        for chunk in affected_chunks:
            chunk.is_dirty = True
            if not chunk.is_meshing:
                chunk.is_meshing = True
                self.request_queue.put(chunk)
        
        
    def modify_terrain(self, world_pos, brush_settings, strength, tool_type, current_action, paint_index=0, target_height=None, target_normal=None, stroke_anchor_pos=None):
        brush_size = brush_settings["size"]
        strength = brush_settings["strength"]
        brush_shape = brush_settings["shape"]
        softness = brush_settings["softness"]

        center_x, center_z = world_pos.x, world_pos.z
        radius = brush_size / 2.0
        affected_chunks = set()
        
        height_updates = []
        
        exponent = 0.25 + (1.0 - softness) * 3.75

        for x in range(int(np.floor(center_x - radius)), int(np.ceil(center_x + radius))):
            for z in range(int(np.floor(center_z - radius)), int(np.ceil(center_z + radius))):
                
                normalized_dist = 1.0

                if brush_shape == 0: # Dairesel Fırça
                    dist_sq = (x - center_x)**2 + (z - center_z)**2
                    if dist_sq > radius**2: continue
                    # Merkezden uzaklığı [0, 1] aralığında normalize et
                    normalized_dist = np.sqrt(dist_sq) / radius

                elif brush_shape == 1: # Kare Fırça
                    dist_x = abs(x - center_x)
                    dist_z = abs(z - center_z)
                    if dist_x > radius or dist_z > radius: continue
                    # Merkezden en uzak eksendeki mesafeyi normalize et
                    normalized_dist = max(dist_x, dist_z) / radius

                base = max(0.0, 1.0 - normalized_dist)
                falloff = pow(base, exponent)
                if falloff <= 0: continue
                
                
                dist_sq = (x - center_x)**2 + (z - center_z)**2
                if dist_sq > radius**2: continue
                
                chunk = self.get_chunk_at_world_pos(x, z)
                if not chunk: continue

                local_x = int(round(x - chunk.position.x * CHUNK_WIDTH))
                local_z = int(round(z - chunk.position.z * CHUNK_DEPTH))

                if 0 <= local_x < CHUNK_WIDTH + 1 and 0 <= local_z < CHUNK_DEPTH + 1:
                    
                    # Değişiklikten önceki değerleri al
                    old_h = chunk.heightmap[local_x, local_z]
                    old_s = chunk.splatmap[local_z, local_x].copy()
                    
                    new_h, new_s = old_h, old_s

                    if tool_type == "paint":
                        target_weights = np.zeros(3); target_weights[paint_index] = 1.0
                        new_weights = old_s + (target_weights - old_s) * strength * falloff
                        new_s = new_weights / np.sum(new_weights)
                        chunk.splatmap[local_z, local_x] = new_s
                        chunk.splatmap_is_dirty = True
                        # Mirror paint across borders
                        if local_x in (0, CHUNK_WIDTH) or local_z in (0, CHUNK_DEPTH):
                            self._mirror_border_update(chunk, local_x, local_z, None, new_s, current_action)
                    
                    elif tool_type == "raise":
                        new_h = old_h + strength * falloff
                        chunk.heightmap[local_x, local_z] = new_h
                        if local_x in (0, CHUNK_WIDTH) or local_z in (0, CHUNK_DEPTH):
                            self._mirror_border_update(chunk, local_x, local_z, new_h, None, current_action)
                    
                    elif tool_type == "lower":
                        new_h = old_h - strength * falloff
                        chunk.heightmap[local_x, local_z] = new_h
                        if local_x in (0, CHUNK_WIDTH) or local_z in (0, CHUNK_DEPTH):
                            self._mirror_border_update(chunk, local_x, local_z, new_h, None, current_action)

                    elif tool_type == "flatten":
                        if target_height is None: continue
                        new_h = old_h + (target_height - old_h) * falloff * strength
                        chunk.heightmap[local_x, local_z] = new_h
                        if local_x in (0, CHUNK_WIDTH) or local_z in (0, CHUNK_DEPTH):
                            self._mirror_border_update(chunk, local_x, local_z, new_h, None, current_action)
                    
                    elif tool_type == "smooth":
                        total_height = 0; neighbor_count = 0
                        for dx in range(-1, 2):
                            for dz in range(-1, 2):
                                nx, nz = local_x + dx, local_z + dz
                                if 0 <= nx < CHUNK_WIDTH + 1 and 0 <= nz < CHUNK_DEPTH + 1:
                                    total_height += chunk.heightmap[nx, nz]; neighbor_count += 1
                        if neighbor_count > 0:
                            average_height = total_height / neighbor_count
                            new_h_calc = old_h + (average_height - old_h) * strength * falloff
                            height_updates.append((chunk, local_x, local_z, old_h, new_h_calc))
                    elif tool_type in ("slope", "memory_slope"):
                        # Parametre isimlerini yeni mantığa göre kullan
                        if target_normal is not None and stroke_anchor_pos is not None and target_normal[1] != 0:
                            target_y = stroke_anchor_pos[1] - \
                                (target_normal[0] * (x - stroke_anchor_pos[0]) + \
                                 target_normal[2] * (z - stroke_anchor_pos[2])) / target_normal[1]
                            
                            new_h = old_h + (target_y - old_h) * strength * falloff
                            chunk.heightmap[local_x, local_z] = new_h
                            if local_x in (0, CHUNK_WIDTH) or local_z in (0, CHUNK_DEPTH):
                                self._mirror_border_update(chunk, local_x, local_z, new_h, None, current_action)
                            
                            
                    if tool_type != "smooth":
                        current_action.record_change(chunk, local_x, local_z, old_h, new_h, old_s, new_s)
                    
                    affected_chunks.add(chunk)

        if tool_type == "smooth":
            for chunk, lx, lz, old_h_smooth, new_h_smooth in height_updates:
                chunk.heightmap[lx, lz] = new_h_smooth
                old_s_smooth = chunk.splatmap[lz, lx].copy()
                current_action.record_change(chunk, lx, lz, old_h_smooth, new_h_smooth, old_s_smooth, old_s_smooth)
                if lx in (0, CHUNK_WIDTH) or lz in (0, CHUNK_DEPTH):
                    self._mirror_border_update(chunk, lx, lz, new_h_smooth, None, current_action)

        for chunk in affected_chunks:
            if tool_type != "paint" and not chunk.is_meshing:
                chunk.is_dirty = True; chunk.is_meshing = True; self.request_queue.put(chunk)

    def draw(self, shader):
        for chunk in self.chunks.values():
            chunk.draw(shader)
            
    def _mirror_border_update(self, chunk, local_x, local_z, new_h, new_s, action):
        """Mirror border vertex changes to neighbor chunks to avoid seams."""
        cx = int(chunk.position.x); cz = int(chunk.position.z)
        neighbors = []
        # Edges
        if local_x == 0:
            neighbors.append(((cx - 1, 0, cz), CHUNK_WIDTH, local_z))
        if local_x == CHUNK_WIDTH:
            neighbors.append(((cx + 1, 0, cz), 0, local_z))
        if local_z == 0:
            neighbors.append(((cx, 0, cz - 1), local_x, CHUNK_DEPTH))
        if local_z == CHUNK_DEPTH:
            neighbors.append(((cx, 0, cz + 1), local_x, 0))
        # Corners
        if local_x == 0 and local_z == 0:
            neighbors.append(((cx - 1, 0, cz - 1), CHUNK_WIDTH, CHUNK_DEPTH))
        if local_x == CHUNK_WIDTH and local_z == 0:
            neighbors.append(((cx + 1, 0, cz - 1), 0, CHUNK_DEPTH))
        if local_x == 0 and local_z == CHUNK_DEPTH:
            neighbors.append(((cx - 1, 0, cz + 1), CHUNK_WIDTH, 0))
        if local_x == CHUNK_WIDTH and local_z == CHUNK_DEPTH:
            neighbors.append(((cx + 1, 0, cz + 1), 0, 0))

        for key, lx2, lz2 in neighbors:
            nb = self.chunks.get(key)
            if nb is None:
                continue
            old_h2 = nb.heightmap[lx2, lz2]
            old_s2 = nb.splatmap[lz2, lx2].copy()
            if new_h is not None:
                nb.heightmap[lx2, lz2] = new_h
            if new_s is not None:
                nb.splatmap[lz2, lx2] = new_s
                nb.splatmap_is_dirty = True
            if action is not None:
                nh2 = nb.heightmap[lx2, lz2]
                ns2 = nb.splatmap[lz2, lx2].copy()
                action.record_change(nb, lx2, lz2, old_h2, nh2, old_s2, ns2)
            nb.is_dirty = True
            if not nb.is_meshing:
                nb.is_meshing = True
                self.request_queue.put(nb)
    def raycast_terrain(self, ray_origin, ray_direction, max_dist=200):
        current_pos = ray_origin.copy()
        # <<< DEĞİŞİKLİK: Adım boyutunu küçülterek hassasiyeti artır >>>
        step_size = 0.2
        for _ in range(int(max_dist / step_size)):
            terrain_height = self.get_height_at_world_pos(current_pos.x, current_pos.z)
            if current_pos.y < terrain_height:
                # <<< DEĞİŞİKLİK: İmleci yüzeyin çok az üzerine yerleştir (Z-fighting önlemi) >>>
                current_pos.y = terrain_height + 0.1
                return current_pos
            current_pos += ray_direction * step_size
        return None

    # --- Persistence helpers ---
    def _chunk_save_path(self, cx, cz):
        return os.path.join(self.save_dir, f"{cx}_{cz}.npz")

    def save_chunk_to_disk(self, chunk):
        try:
            cx = int(chunk.position.x); cz = int(chunk.position.z)
            path = self._chunk_save_path(cx, cz)
            np.savez_compressed(path, heightmap=chunk.heightmap, splatmap=chunk.splatmap)
        except Exception as e:
            print(f"Chunk kaydetme hatasi: {e}")

    def try_load_chunk_from_disk(self, cx, cz, chunk):
        try:
            path = self._chunk_save_path(cx, cz)
            if os.path.exists(path):
                with np.load(path) as data:
                    if 'heightmap' in data: chunk.heightmap[:, :] = data['heightmap']
                    if 'splatmap' in data: chunk.splatmap[:, :, :] = data['splatmap']
                chunk.splatmap_is_dirty = True
                return True
        except Exception as e:
            print(f"Chunk yukleme hatasi: {e}")
        return False




