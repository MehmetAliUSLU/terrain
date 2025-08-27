# history.py

class EditAction:
    """
    Tek bir düzenleme eylemini (farenin basılı tutulması) temsil eder.
    Değişen tüm noktaların eski ve yeni durumlarını saklar.
    """
    def __init__(self, chunk):
        self.chunk = chunk
        # Değişiklikleri saklamak için sözlükler (dictionary)
        # Anahtar: (local_x, local_z), Değer: ilgili veri
        self.old_heights = {}
        self.new_heights = {}
        self.old_splats = {}
        self.new_splats = {}

    def record_change(self, lx, lz, old_h, new_h, old_s, new_s):
        """Bir noktanın değişimini kaydeder."""
        key = (lx, lz)
        # Sadece ilk "eski" durumu kaydet
        if key not in self.old_heights:
            self.old_heights[key] = old_h
            self.old_splats[key] = old_s
        # Her zaman en son "yeni" durumu güncelle
        self.new_heights[key] = new_h
        self.new_splats[key] = new_s

    def is_empty(self):
        """Eylemin herhangi bir değişiklik kaydedip kaydetmediğini kontrol eder."""
        return not self.new_heights and not self.new_splats

    def apply_changes(self, height_data, splat_data):
        """Verilen verileri chunk'a uygular."""
        for (lx, lz), height in height_data.items():
            self.chunk.heightmap[lx, lz] = height
        
        for (lx, lz), splat in splat_data.items():
            self.chunk.splatmap[lz, lx] = splat
        
        # DÜZELTME: is_meshing bayrağını burada ayarlama!
        # Sadece chunk'ın güncellenmesi gerektiğini işaretle.
        self.chunk.is_dirty = True
        self.chunk.splatmap_is_dirty = True

    def undo(self):
        """Eylemi geri alır."""
        print("Geri Alındı")
        self.apply_changes(self.old_heights, self.old_splats)

    def redo(self):
        """Eylemi yineler."""
        print("Yinelendi")
        self.apply_changes(self.new_heights, self.new_splats)


class UndoManager:
    """Geri alma ve yineleme eylemlerini yönetir."""
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []

    def register_action(self, action):
        """Yeni bir eylemi kaydeder."""
        if action.is_empty():
            return
        self.undo_stack.append(action)
        self.redo_stack.clear() # Yeni bir eylem yapıldığında yineleme geçmişi silinir
        print(f"Eylem Kaydedildi. Geri Alma Yığını Boyutu: {len(self.undo_stack)}")

    def undo(self):
        """Son eylemi geri alır."""
        if not self.undo_stack:
            print("Geri alınacak eylem yok.")
            return
        action = self.undo_stack.pop()
        action.undo()
        self.redo_stack.append(action)

    def redo(self):
        """Son geri alınan eylemi yineler."""
        if not self.redo_stack:
            print("Yinelenecek eylem yok.")
            return
        action = self.redo_stack.pop()
        action.redo()
        self.undo_stack.append(action)