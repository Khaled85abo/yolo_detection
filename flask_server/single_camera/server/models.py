from typing import List, Tuple

class PlankStatus:
    def __init__(self):
        self.overlap: List[Tuple[int, int]] = []
        self.stop: List[int] = []
        self.incorrect: List[int] = []
        self.conveyor_stop: bool = False
    
    def to_dict(self):
        """Convert status to dictionary for API responses"""
        return {
            'overlap': len(self.overlap) > 0,
            'stop': len(self.stop) > 0,
            'incorrect': len(self.incorrect) > 0,
            'conveyor_stop': self.conveyor_stop
        }
    
    def update(self, overlapped=None, stopped=None, incorrect=None):
        """Update status values and return if status changed"""
        status_changed = False
        
        if overlapped is not None:
            sorted_overlapped = sorted(overlapped)
            if sorted_overlapped != self.overlap:
                self.overlap = sorted_overlapped
                status_changed = True
        
        if stopped is not None:
            sorted_stopped = sorted(stopped)
            if sorted_stopped != self.stop:
                self.stop = sorted_stopped
                status_changed = True
        
        if incorrect is not None:
            sorted_incorrect = sorted(incorrect)
            if sorted_incorrect != self.incorrect:
                self.incorrect = sorted_incorrect
                status_changed = True
                
        return status_changed