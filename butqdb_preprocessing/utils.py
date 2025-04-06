import json 
import re

def find_inner_segments(lookup_table, record, annotator, segstart, segend):
    # Initialize the list to hold the output (start, end, class)
    class_ranges = []
    
    # Iterate through the class ranges in the lookup table for the specific record
    for start, end, score in lookup_table[record][annotator]:
        # Check if the segment overlaps with the range (segstart, segend)
        if start <= segend and end >= segstart:
            # Calculate the overlap range
            overlap_start = max(start, segstart)
            overlap_end = min(end, segend)
            
            # Append the tuple of (start, end, class) for the overlapping range
            class_ranges.append((overlap_start, overlap_end, int(score)))
    return class_ranges

def load_json(path_to_file):
    with open(path_to_file, "r") as f:
        data = json.load(f)
    return data

def sample_to_time(sample_idx):
    '''
    Converts number or array of sample indices to a string or list of strings in the format hh:mm:ss.
    Uses hard-coded 1000Hz sampling rate.
    '''
    if hasattr(sample_idx, '__iter__'):
        out = [f"{int(v // 3600000):02}:{int((v % 3600000) // 60000):02}:{int((v % 60000) // 1000):02}" for v in sample_idx]
    else:
        out =  f"{int(sample_idx // 3600000):02}:{int((sample_idx % 3600000) // 60000):02}:{int((sample_idx % 60000) // 1000):02}"
    return out

def time_to_sample(time):
    """
    Converts a time string or a list of time strings in the format hh:mm:ss to sample indices.
    Uses hard-coded 1000Hz sampling rate.
    """
    if hasattr(time, '__iter__') and not isinstance(time, str):
        out = [int(t.split(':')[0]) * 3600000 + int(t.split(':')[1]) * 60000 + int(t.split(':')[2]) * 1000 for t in time]
    else:
        out = int(time.split(':')[0]) * 3600000 + int(time.split(':')[1]) * 60000 + int(time.split(':')[2]) * 1000
    return out


class Interval:
    def __init__(self, segments=None):
        # stores segments as tuples: (start, end, class)
        self.segments = []
        if segments:
            if isinstance(segments, list):
                # assume list contains tuple segments
                self.segments.extend(segments)
            elif isinstance(segments, tuple) and len(segments) == 3:
                start, end, class_id = segments
                self.segments.append((start, end, class_id))
            else:
                raise ValueError("Initialize with a list of segment tuples or a single segment tuple.")
            # sort segments
            self.segments.sort(key=lambda seg: seg[0])

    def add_segment(self, segment):
        """
        Add a new segment to the timeline.
        Assumes the input segments do not overlap.
        """
        start, end, class_id = segment
        self.segments.append((start, end, class_id))
        self.segments.sort()

    def replace_range(self, segment):
        """
        Replace a range (start, end) with a new class.
        Splits and modifies the timeline accordingly.
        """
        start, end, new_class = segment
        new_segments = []

        # Flag to check if replacement has been added
        replacement_added = False

        for seg_start, seg_end, seg_class in self.segments:
            # If the current segment ends before the replacement range starts, keep it
            if seg_end < start:
                new_segments.append((seg_start, seg_end, seg_class))
            # If the current segment starts after the replacement range ends, keep it
            elif seg_start > end:
                new_segments.append((seg_start, seg_end, seg_class))
            else:
                # Handle overlap: split the segment into parts
                if seg_start < start:
                    # Left part of the current segment remains unchanged
                    new_segments.append((seg_start, start - 1, seg_class))
                if seg_end > end:
                    # Right part of the current segment remains unchanged
                    new_segments.append((end + 1, seg_end, seg_class))

                # Add the replacement segment only once
                if not replacement_added:
                    new_segments.append((start, end, new_class))
                    replacement_added = True

        # Now we need to merge adjacent segments with the same class
        merged_segments = []
        for seg_start, seg_end, seg_class in sorted(new_segments):
            if merged_segments and merged_segments[-1][2] == seg_class and merged_segments[-1][1] + 1 >= seg_start:
                # Merge the current segment with the last segment if they are of the same class and adjacent
                last_start, last_end, last_class = merged_segments[-1]
                merged_segments[-1] = (last_start, max(last_end, seg_end), last_class)
            else:
                # No merge needed, just add the current segment
                merged_segments.append((seg_start, seg_end, seg_class))
        self.segments = merged_segments
                
    def get_subsegments(self, seg_start, seg_end):
        inners = []
        for start, end, score in self.segments:
            if start <= seg_end and end >= seg_start:
                overlap_start = max(start, seg_start)
                overlap_end = min(end, seg_end)
                inners.append((overlap_start, overlap_end, int(score)))
        return inners
        
    def __iter__(self):
        """
        Makes the timeline iterable by returning an iterator over its segments.
        """
        return iter(self.segments)
    
    def __str__(self):
        """
        String representation of the timeline for easy visualization.
        """
        return "\n".join(f"({start}, {end}, {class_id})" for start, end, class_id in self.segments)