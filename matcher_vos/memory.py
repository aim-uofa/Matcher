import math
import heapq

class Frame():
    def __init__(
        self,
        img_np,
        feat,
        frame_id,
        mask,
        mask_score
    ):
        self.mask_score_decayed = mask_score
        self.img_np = img_np
        self.feat = feat
        self.frame_id = frame_id
        self.mask = mask
        self.mask_score = mask_score
        
    def __lt__(self, other):
        a = (
            self.mask_score_decayed,
            self.img_np,
            self.feat,
            self.frame_id,
            self.mask,
            self.mask_score
        )
        b = (
            other.mask_score_decayed,
            other.img_np,
            other.feat,
            other.frame_id,
            other.mask,
            other.mask_score
        )
        return a < b
        
class Memory():
    def __init__(
        self,
        memory_len = 1,
        fix_last_frame = False,
        memory_decay_ratio = 20,
        memory_decay_type = 'cos'
    ):
        self.memory_len = memory_len - fix_last_frame
        assert self.memory_len >= 0
        self.memory = []
        self.fix_last_frame = fix_last_frame
        self.last_frame = None
        self.mask_score_decay = memory_decay_ratio

        if memory_decay_ratio:
            if memory_decay_type == 'cos':
                self.mask_score_decay_table = [max(0, math.cos(x/memory_decay_ratio)) for x in range(15)]
            elif memory_decay_type == 'linear':
                self.mask_score_decay_table = [max(0, 1-x/memory_decay_ratio) for x in range(int(memory_decay_ratio))]
            elif memory_decay_type == 'ellipse':
                self.mask_score_decay_table = [max(0, (1-(x/memory_decay_ratio)**2)**0.5) for x in range(int(memory_decay_ratio))]
            elif memory_decay_type == 'exp':
                self.mask_score_decay_table = [max(0, math.exp(-x/memory_decay_ratio)) for x in range(15)]
            elif memory_decay_type == 'constant':
                self.mask_score_decay_table = [1] * 15
                

    def _get_mask_score_decay_ratio(self, x):
        if not self.mask_score_decay:
            return 1
        elif x < len(self.mask_score_decay_table):
            return self.mask_score_decay_table[x]
        else:
            return 0
        
    def update_memory(self, frame):
        if self.fix_last_frame:
            if self.last_frame is not None:
                heapq.heappush(self.memory, self.last_frame)
            self.last_frame = frame
        else:
            heapq.heappush(self.memory, frame)
            
        for i in range(len(self.memory)):
            mask_score_decay_ratio = self._get_mask_score_decay_ratio(frame.frame_id - self.memory[i].frame_id)
            mask_score_decayed = mask_score_decay_ratio * self.memory[i].mask_score
            self.memory[i].mask_score_decayed = mask_score_decayed
        heapq.heapify(self.memory)
        if len(self.memory) > self.memory_len:
            heapq.heappop(self.memory)
    
    def get_memory(self):
        if self.last_frame is not None:
            return self.memory + [self.last_frame]
        else:
            return self.memory
        
    def clear_memory(self):
        self.last_frame = None
        self.memory = []