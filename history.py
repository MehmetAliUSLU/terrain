# history.py

class EditAction:
    """
    Tek bir düzenleme eylemini (farenin basılı tutulması) temsil eder.
    Birden çok chunk üzerindeki değişiklikleri tek eylemde toplar.
    """
    def __init__(self):
        # {chunk: {(lx,lz): value}}
        self.old_heights = {}
        self.new_heights = {}
        self.old_splats = {}
        self.new_splats = {}

    def _ensure(self, chunk):
        if chunk not in self.old_heights:
            self.old_heights[chunk] = {}
        if chunk not in self.new_heights:
            self.new_heights[chunk] = {}
        if chunk not in self.old_splats:
            self.old_splats[chunk] = {}
        if chunk not in self.new_splats:
            self.new_splats[chunk] = {}

    def record_change(self, chunk, lx, lz, old_h, new_h, old_s, new_s):
        key = (lx, lz)
        self._ensure(chunk)
        if key not in self.old_heights[chunk]:
            self.old_heights[chunk][key] = old_h
            self.old_splats[chunk][key] = old_s
        self.new_heights[chunk][key] = new_h
        self.new_splats[chunk][key] = new_s

    def is_empty(self):
        any_new_height = any(len(d) > 0 for d in self.new_heights.values())
        any_new_splat = any(len(d) > 0 for d in self.new_splats.values())
        return not (any_new_height or any_new_splat)

    def _apply(self, use_new: bool):
        chunks = set(self.old_heights.keys()) | set(self.new_heights.keys()) | set(self.old_splats.keys()) | set(self.new_splats.keys())
        for chunk in chunks:
            heights = self.new_heights.get(chunk, {}) if use_new else self.old_heights.get(chunk, {})
            splats = self.new_splats.get(chunk, {}) if use_new else self.old_splats.get(chunk, {})
            for (lx, lz), h in heights.items():
                chunk.heightmap[lx, lz] = h
            for (lx, lz), s in splats.items():
                chunk.splatmap[lz, lx] = s
            chunk.is_dirty = True
            if len(splats) > 0:
                chunk.splatmap_is_dirty = True

    def undo(self):
        print("Geri Alındı")
        self._apply(use_new=False)

    def redo(self):
        print("Yinelendi")
        self._apply(use_new=True)


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
        self.redo_stack.clear()  # Yeni bir eylem yapıldığında yineleme geçmişi silinir
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

