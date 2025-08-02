import numpy as np
import pygame
from collections import deque, defaultdict
from typing import List

from color import normalize_weights, ColorMap
from ring_lab import PigeonRingLab
from OpenGL.GL import (
    glGenTextures, glBindTexture, glTexParameteri, glTexImage2D, glTexSubImage2D,
    glBegin, glEnd, glTexCoord2f, glVertex2f, glClear, glClearColor, glViewport,
    glLoadIdentity, glOrtho, glMatrixMode, glEnable,
    GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE, GL_LINEAR, GL_CLAMP_TO_EDGE,
    GL_QUADS, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_PROJECTION,
    GL_MODELVIEW, GL_NEAREST,
)


class PigeonVisualizer:
    def __init__(
        self,
        lab: PigeonRingLab,
        win_width: int = 960,
        win_height: int = 540,
        max_fps: int = 144,
        feedback_delay_frames: int = 0,
        input_hue_tape_depth: int = 0,
        bit_depth: int = 16,
        row_height_gamma: float = 2.0,
    ):
        pygame.init()
        pygame.display.set_caption("Pigeon Crushing Visualizer")
        self.max_fps = max_fps
        self.win_width = win_width
        self.win_height = win_height
        self.feedback_delay_frames = feedback_delay_frames
        self.input_hue_tape_depth = input_hue_tape_depth
        self.output_hue_tape_depth = -1
        self.MAX_DISPLAY_WIDTH = win_width
        self.bit_depth = bit_depth
        self.lab = lab
        self.jobs = lab.ring_jobs
        self.num_jobs = len(self.jobs)
        pygame.display.set_mode(
            (self.win_width, self.win_height),
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.HWSURFACE
        )
        self._init_gl()
        self.in_textures = [self._make_texture(min(job.input_bins, self.MAX_DISPLAY_WIDTH)) for job in self.jobs]
        self.out_textures = [self._make_texture(min(job.output_bins, self.MAX_DISPLAY_WIDTH)) for job in self.jobs]
        self.feedback_buffer = deque()
        # Use results and job_manager from lab
        self.results = lab.results
        self.job_manager = lab.job_manager

    def _init_gl(self):
        glViewport(0, 0, self.win_width, self.win_height)
        glClearColor(0.07, 0.07, 0.07, 1.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.win_width, self.win_height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_TEXTURE_2D)

    def _make_texture(self, width: int):
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        empty = np.zeros((1, width, 4), dtype=np.uint8)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, empty)
        return tex_id

    def _calculate_dynamic_row_heights(self, total_height: int) -> List[int]:
        N = len(self.jobs)
        focus = getattr(self, "row_focus", 0.5)
        raw_weights = []
        for i, job in enumerate(self.jobs):
            pos = i / max(N-1, 1)
            focus_curve = (1 - abs(pos - focus) * 2) ** 2.0 if focus != 0.5 else 1.0
            linear_weight = (min(job.input_bins, self.MAX_DISPLAY_WIDTH) / job.input_bins +
                            min(job.output_bins, self.MAX_DISPLAY_WIDTH) / job.output_bins) / 2.0
            gamma_weight = linear_weight ** 2.0
            final_weight = 0.1 * (1 - gamma_weight) + 3.0 * gamma_weight
            final_weight *= focus_curve
            raw_weights.append(final_weight)
        total_raw_weight = sum(raw_weights)
        normalized_weights = [w / total_raw_weight for w in raw_weights] if total_raw_weight > 0 else [1.0/N]*N
        return normalize_weights(normalized_weights, total_height)

    def update_textures(self, job_idx: int, in_rgba: np.ndarray, out_rgba: np.ndarray):
        in_tex_width = in_rgba.shape[1]
        glBindTexture(GL_TEXTURE_2D, self.in_textures[job_idx])
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, in_tex_width, 1, GL_RGBA, GL_UNSIGNED_BYTE, in_rgba)
        out_tex_width = out_rgba.shape[1]
        glBindTexture(GL_TEXTURE_2D, self.out_textures[job_idx])
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, out_tex_width, 1, GL_RGBA, GL_UNSIGNED_BYTE, out_rgba)

    def draw_texture_row(self, tex_id: int, bins: int, x: int, y: int, width: int, height: int):
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(x, y)
        glTexCoord2f(1, 0)
        glVertex2f(x + width, y)
        glTexCoord2f(1, 1)
        glVertex2f(x + width, y + height)
        glTexCoord2f(0, 1)
        glVertex2f(x, y + height)
        glEnd()

    def run(self):
        clock = pygame.time.Clock()
        running = True
        self.job_manager.start()
        controls_printed = False

        while running:
            if not controls_printed:
                print("\n--- Row Height Controls ---")
                print(" Min Height (Q/A) | Max Height (W/S) | Gamma (E/D)")
                controls_printed = True

            # --- Event Handling for Live Controls ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q: self.min_row_weight += 0.05
                    if event.key == pygame.K_a: self.min_row_weight = max(0, self.min_row_weight - 0.05)
                    if event.key == pygame.K_w: self.max_row_weight += 0.1
                    if event.key == pygame.K_s: self.max_row_weight = max(0, self.max_row_weight - 0.1)
                    if event.key == pygame.K_e: self.row_height_gamma += 0.1
                    if event.key == pygame.K_d: self.row_height_gamma = max(0.1, self.row_height_gamma - 0.1)
                    
                    print(f"\rMin: {self.min_row_weight:.2f}, Max: {self.max_row_weight:.2f}, Gamma: {self.row_height_gamma:.2f}", end="")
            keys = pygame.key.get_pressed()
            change = False
            if keys[pygame.K_q]:
                self.min_row_weight += 0.01
                change = True
            if keys[pygame.K_a]:
                self.min_row_weight = max(0, self.min_row_weight - 0.01)
                change = True
            if keys[pygame.K_w]:
                self.max_row_weight += 0.02
                change = True
            if keys[pygame.K_s]:
                self.max_row_weight = max(0, self.max_row_weight - 0.02)
                change = True
            if keys[pygame.K_e]:
                self.row_height_gamma += 0.01
                change = True
            if keys[pygame.K_d]:
                self.row_height_gamma = max(0.1, self.row_height_gamma - 0.01)
                change = True
            if change:
                print(f"\rMin: {self.min_row_weight:.2f}, Max: {self.max_row_weight:.2f}, Gamma: {self.row_height_gamma:.2f}", end="")

            # --- Feedback logic ---
            if (self.num_jobs - 1) in self.results:
                bins, _ = self.results[self.num_jobs - 1]
                feedback_payloads = []
                for bin_idx, bin_list in bins.items():
                    for payload in bin_list:
                        feedback_payloads.append((bin_idx, payload))
                self.feedback_buffer.append(feedback_payloads)
                if len(self.feedback_buffer) > self.feedback_delay_frames:
                    delayed_payloads = self.feedback_buffer.popleft()
                    self.jobs[0].source = iter(delayed_payloads)

            # --- Drawing Logic ---
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            surface = pygame.display.get_surface()
            window_w, window_h = surface.get_size()
            
            row_heights = self._calculate_dynamic_row_heights(window_h)

            half_w = window_w // 2
            y = 0 
            for idx, job in enumerate(self.jobs):
                current_row_height = row_heights[idx]
                
                # (Data gathering logic)
                bins = []
                in_hues = job.in_hues
                out_hues = []
                if idx in self.results:
                    bins, df = self.results[idx]
                    if self.input_hue_tape_depth is not None:
                        input_bin_payloads = defaultdict(list)
                        prev_job_idx = (idx - 1 + self.num_jobs) % self.num_jobs
                        if prev_job_idx in self.results:
                            prev_bins, _ = self.results[prev_job_idx]
                            for bin_idx, bin_list in enumerate(prev_bins):
                                for payload in bin_list:
                                    input_bin_payloads[bin_idx].append(payload)
                        new_in_hues = []
                        for i in range(job.input_bins):
                            payloads = input_bin_payloads[i]
                            if payloads:
                                hues = [p.tape_color(depth=self.input_hue_tape_depth, mode='indexed') for p in payloads]
                                new_in_hues.append(sum(hues) / len(hues))
                            else:
                                new_in_hues.append(ColorMap.fallback_hue(i, "input"))
                        in_hues = new_in_hues
                    
                    final_bins = [bins.get(i, []) for i in range(job.output_bins)]
                    for i, bin_list in enumerate(final_bins):
                        if bin_list:
                            hues = [p.tape[self.output_hue_tape_depth] for p in bin_list if p.tape]
                            out_hue = sum(hues) / len(hues) if hues else job.out_hues[i]
                        else:
                            out_hue = ColorMap.fallback_hue(i, "output")
                        out_hues.append(out_hue)
                else:
                    out_hues = job.out_hues
 
                # (Interpolation and drawing logic)
                in_rgba = ColorMap.interpolate_hues(in_hues, min(job.input_bins, self.MAX_DISPLAY_WIDTH))
                out_rgba = ColorMap.interpolate_hues(out_hues, min(job.output_bins, self.MAX_DISPLAY_WIDTH))
                self.update_textures(idx, in_rgba, out_rgba)
    
                in_display_width = min(job.input_bins, self.MAX_DISPLAY_WIDTH)
                out_display_width = min(job.output_bins, self.MAX_DISPLAY_WIDTH)
                self.draw_texture_row(self.in_textures[idx], in_display_width, 0, y, half_w, current_row_height)
                self.draw_texture_row(self.out_textures[idx], out_display_width, half_w, y, window_w - half_w, current_row_height)
                
                y += current_row_height

            pygame.display.flip()
            clock.tick(self.max_fps)
        pygame.quit()

