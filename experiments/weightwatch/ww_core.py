# ruff: noqa
# pyright: basic, reportOptionalIterable=false, reportOptionalSubscript=false, reportArgumentType=false, reportCallIssue=false
"""WeightWatch's detector core, copied verbatim (P2.73). Never edited by hand.

Source: github.com/fjzzq2002/WeightWatch at 8d3874fb90a0aad5478514f96d71779d2aedf5aa, `src/utils.py` (sha256
a7e2a3ee139faeda2941e38d8923fe7602ddf3b3c6f6f482c490067b6bcd5792), cloned read only in artifacts/p273_src/WeightWatch.
Copied: line 6 (`EPS`) and lines 156-343 (class `LatentStats`: `__init__`, `register`, `compactify`, `uncompactify`,
`set_threshold`), unchanged but for one storage-only patch, at source line 245:
    self.datapoints.append(np.array(current_datapoint))  ->  self.datapoints.append(current_datapoint)
NumPy 2.4.0 (pinned in pyproject.toml) refuses `np.array` of a ragged list, which every item with layers of unequal
direction counts makes; `set_threshold` iterates the stored list unchanged. The file's top imports (transformers, peft)
are not copied: the class uses only torch and numpy. `ww_detector.check_core` checks this file against the source text
(read as text, never imported) at stage 0, and the test does the same."""
import torch
import numpy as np

EPS = 0.01


class LatentStats:
    def __init__(self, max_layers=33, us=None, us_desc=None, no_record=False):
        max_x = max_layers + 1  # +1 for flexibility
        max_y = 80  # Should be enough for directions
        self.maxima = np.full((max_x, max_y), -float('inf'))
        self.minima = np.full((max_x, max_y), float('inf'))
        self.maxima_prompt = [['' for _ in range(max_y)] for _ in range(max_x)]
        self.minima_prompt = [['' for _ in range(max_y)] for _ in range(max_x)]
        self.ranges = [[[] for _ in range(max_y)] for _ in range(max_x)]
        self.recorded = [[[] for _ in range(max_y)] for _ in range(max_x)]
        self.num_samples = 0
        self.datapoints = []
        self.datapoint_prompts = []
        self.us = us
        self.us_desc = us_desc
        # no_record: if True, do not record datapoints, and thus do not allow for threshold recalculation
        self.no_record = no_record

    @torch.inference_mode()
    def register(self, latent, prompt, read_only=False, record_all=False, use_norms=False, use_norm_layerid=None, verbose=False):
        assert self.us is not None and self.us_desc is not None

        if len(latent) == 0:
            return False
            
        if not read_only:
            self.num_samples += latent.shape[0]
        triggered = False
        
        # Store current datapoint ranges for threshold recalculation
        current_datapoint = []
        
        for ix, ux in enumerate(self.us):
            if len(ux) == 0:
                current_datapoint.append([])
                continue

            li = latent[:, ix]
            # normalize
            li = li / li.norm(dim=-1, keepdim=True)
            sims = ux @ li.T  # cosine similarity; assume ux is 
            sims_mx = torch.max(sims, dim=-1).values
            sims_mi = torch.min(sims, dim=-1).values
            sims_sum = torch.sum(sims, dim=-1)
            sims_sum_sq = torch.sum(sims**2, dim=-1)
            sims_mx = sims_mx.tolist()
            sims_mi = sims_mi.tolist()
            sims_sum = sims_sum.tolist()
            sims_sum_sq = sims_sum_sq.tolist()

            # Initialize layer data for current datapoint
            layer_ranges = []
            
            for uid, (sim_mx, sim_mi, sim_sum, sim_sum_sq, ss) in enumerate(zip(sims_mx, sims_mi, sims_sum, sims_sum_sq, sims)):
                if uid >= len(self.us_desc[ix]):
                    break
                    
                def trigger(x):
                    if self.minima[ix][uid] - EPS <= x <= self.maxima[ix][uid] + EPS:
                        return False
                    return True
                
                triggered |= trigger(sim_mx) or trigger(sim_mi)
                
                if read_only:
                    if triggered:
                        return True
                    continue
                
                if not self.no_record:
                    if record_all:
                        self.recorded[ix][uid].extend(ss.tolist())
                    self.ranges[ix][uid].append((sim_mi, sim_mx))
                    # Store range for current datapoint
                    layer_ranges.append((sim_mi, sim_mx))
                
                # Update min/max
                if sim_mx > self.maxima[ix][uid]:
                    self.maxima[ix][uid] = sim_mx
                    self.maxima_prompt[ix][uid] = prompt
                if sim_mi < self.minima[ix][uid]:
                    self.minima[ix][uid] = sim_mi
                    self.minima_prompt[ix][uid] = prompt
            
            current_datapoint.append(layer_ranges)
        
        # Store the current datapoint if not read_only
        if not read_only and not self.no_record:
            self.datapoint_prompts.append(prompt)
            self.datapoints.append(current_datapoint)
        
        return triggered
    
    def compactify(self):
        for i in range(len(self.ranges)):
            for j in range(len(self.ranges[i])):
                self.ranges[i][j] = np.array(self.ranges[i][j])
    
    def uncompactify(self):
        for i in range(len(self.ranges)):
            for j in range(len(self.ranges[i])):
                self.ranges[i][j] = list(self.ranges[i][j])  # unzip

    def set_threshold(self, percentile, verbose=False, return_details=False):
        """
        Recalculate maxima and minima by dropping percentile from both ends.
        Returns the false positive rate on existing data.
        
        Args:
            percentile: float, e.g., 0.01 for 1%
        
        Returns:
            false_positive_rate: float
        """
        if verbose:
            print(f"Setting threshold with {percentile*100:.5f}% percentile trimming...")

        triggered = []
        
        for val in [1,0]:
            # Recalculate thresholds
            new_maxima = np.full_like(self.maxima, -float('inf'))
            new_minima = np.full_like(self.minima, float('inf'))

            val_split = 1 if val == 0 else int(len(self.datapoints) * 0.1)
            
            for ix in range(len(self.ranges)):
                for uid in range(len(self.ranges[ix])):
                    if len(self.ranges[ix][uid]) == 0:
                        continue
                    
                    # Get all min and max values for this layer/direction
                    all_mins = [r[0] for r in self.ranges[ix][uid][:-val_split]]
                    all_maxs = [r[1] for r in self.ranges[ix][uid][:-val_split]]

                    all_mins = sorted(all_mins)
                    all_maxs = sorted(all_maxs)
                    
                    if len(all_mins) > 0:
                        # Calculate percentiles
                        min_threshold = np.percentile(all_mins, percentile * 100)
                        max_threshold = np.percentile(all_maxs, (1 - percentile) * 100)

                        # check first 10 of allmins
                        # print(f'{all_mins[:10]=} {all_maxs[:10]=} {all_mins[-10:]=} {all_maxs[-10:]=}')
                        # print(f'{len(all_mins)=} {len(all_maxs)=} {min_threshold=} {max_threshold=}')
                        
                        new_minima[ix][uid] = min_threshold
                        new_maxima[ix][uid] = max_threshold
            
            # Update the thresholds
            self.maxima = new_maxima
            self.minima = new_minima

            print(f'{val_split=} {len(self.datapoints)=}',flush=True)

            if val == 0:
                break
            
            # Calculate false positive rate on validation split
            flagged_datapoints = 0
            
            if val_split > 0:
                for datapoint in self.datapoints[-val_split:]:
                    is_flagged = False
                    for ix, layer_ranges in enumerate(datapoint):
                        for uid, (sim_mi, sim_mx) in enumerate(layer_ranges):
                            if ix < len(self.us) and uid < len(self.us[ix]):
                                mi, mx = self.minima[ix][uid], self.maxima[ix][uid]
                                if sim_mi < mi - EPS or sim_mx > mx + EPS:
                                    is_flagged = True
                                    break
                        if is_flagged:
                            break
                    if is_flagged:
                        flagged_datapoints += 1
                    triggered.append(is_flagged)
                
                false_positive_rate = flagged_datapoints / min(len(self.datapoints), val_split)
            else:
                false_positive_rate = 0.0
        
            if verbose:
                print(f"New thresholds set. False positive rate: {false_positive_rate:.4f} ({flagged_datapoints}/{min(len(self.datapoints), val_split)})")
        if return_details:
            return false_positive_rate, np.array(triggered)
        else:
            return false_positive_rate
