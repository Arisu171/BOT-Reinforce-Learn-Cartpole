const el = (id) => document.getElementById(id);

    // Waveform history buffer
    const MAX_HISTORY = 150;
    const historyAngles = [];
    const historyPositions = [];

    // Categorized models cache
    let modelsData = { dqn: [], reinforce: [], all: [] };

    // Canvas setup
    const canvas = el('waveform-canvas');
    const ctx = canvas.getContext('2d');

    const resizeCanvas = () => {
      canvas.width = canvas.parentElement.clientWidth;
      canvas.height = canvas.parentElement.clientHeight;
    };
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    const drawWaveform = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const w = canvas.width;
      const h = canvas.height;

      // Draw center baseline
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();
      ctx.setLineDash([]);

      if (historyAngles.length < 2) return;

      const stepX = w / (MAX_HISTORY - 1);

      // Draw Position x (cyan)
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < historyPositions.length; i++) {
        const x = i * stepX;
        const norm = (historyPositions[i] + 2.4) / 4.8;
        const y = h - norm * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Draw Angle theta (emerald)
      ctx.strokeStyle = '#34d399';
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < historyAngles.length; i++) {
        const x = i * stepX;
        const norm = (historyAngles[i] + 12.0) / 24.0;
        const y = h - norm * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    // Format uptime helper
    const formatUptime = (totalSec) => {
      const s = Math.floor(totalSec % 60);
      const m = Math.floor((totalSec / 60) % 60);
      const h = Math.floor(totalSec / 3600);
      return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    };

    // Update Telemetry UI
    const updateTelemetry = (data) => {
      if (!data || !data.state) return;

      el('uptime-val').textContent = formatUptime(data.uptime_seconds || 0);
      el('total-steps-val').textContent = (data.total_steps || 0).toLocaleString();
      el('fps-val').textContent = (data.fps_real || 30.0).toFixed(1);

      // State readouts
      const deg = data.state.theta_deg || 0;
      const x = data.state.x || 0;
      el('angle-val').textContent = `${deg.toFixed(2)}°`;
      el('pos-val').textContent = `${x.toFixed(3)} m`;
      el('xdot-val').textContent = `${(data.state.x_dot || 0).toFixed(3)} m/s`;
      el('thetadot-val').textContent = `${(data.state.theta_dot || 0).toFixed(3)} rad/s`;

      // Warning color for angle
      if (Math.abs(deg) > 8.0) {
        el('angle-val').className = 'font-mono text-red-400 font-bold';
      } else if (Math.abs(deg) > 4.0) {
        el('angle-val').className = 'font-mono text-amber-400 font-bold';
      } else {
        el('angle-val').className = 'font-mono text-emerald-400 font-bold';
      }

      // Needle positions (0..100%)
      const anglePct = Math.min(100, Math.max(0, ((deg + 12.0) / 24.0) * 100));
      const posPct = Math.min(100, Math.max(0, ((x + 2.4) / 4.8) * 100));
      el('angle-needle').style.left = `${anglePct}%`;
      el('pos-needle').style.left = `${posPct}%`;

      // Waveform push
      historyAngles.push(deg);
      historyPositions.push(x);
      if (historyAngles.length > MAX_HISTORY) {
        historyAngles.shift();
        historyPositions.shift();
      }
      drawWaveform();

      // Action and dynamic balance display
      const isRight = data.action === 1;
      el('action-display').textContent = isRight ? 'PUSH RIGHT (1)' : 'PUSH LEFT (0)';
      el('action-display').className = isRight 
        ? 'px-2.5 py-1 rounded-lg font-bold bg-sky-500/20 text-sky-400 border border-sky-500/30'
        : 'px-2.5 py-1 rounded-lg font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';

      if (data.action_ratio) {
        el('balance-ratio').textContent = `${data.action_ratio.left}% L / ${data.action_ratio.right}% R`;
      }

      if (data.probabilities) {
        const pLeft = Number(data.probabilities[0]);
        const pRight = Number(data.probabilities[1]);
        el('prob-left').textContent = pLeft.toFixed(2);
        el('prob-right').textContent = pRight.toFixed(2);
        el('prob-bar-left').style.width = `${Math.min(100, Math.max(0, pLeft * 100))}%`;
        el('prob-bar-right').style.width = `${Math.min(100, Math.max(0, pRight * 100))}%`;
      }

      if (data.q_values) {
        el('q-left-val').textContent = Number(data.q_values[0]).toFixed(2);
        el('q-right-val').textContent = Number(data.q_values[1]).toFixed(2);
      } else {
        el('q-left-val').textContent = 'N/A';
        el('q-right-val').textContent = 'N/A';
      }

      // Badges
      el('badge-algo').textContent = data.algo.toUpperCase();
      el('badge-mode').textContent = data.mode === 'infinite' ? 'Infinite Horizon' : 'Standard 500 Steps';

      // Session Metrics
      el('stat-ep-steps').textContent = (data.episode_steps || 0).toLocaleString();
      el('stat-best-steps').textContent = (data.best_steps || 0).toLocaleString();
      el('stat-total-ep').textContent = (data.total_episodes || 0).toLocaleString();

      if (data.recent_episodes && data.recent_episodes.length > 0) {
        const sum = data.recent_episodes.reduce((acc, ep) => acc + ep.steps, 0);
        const avg = sum / data.recent_episodes.length;
        el('stat-avg-score').textContent = avg.toFixed(1);

        const listContainer = el('recent-episodes-list');
        listContainer.innerHTML = data.recent_episodes.slice().reverse().map(ep => `
          <div class="flex justify-between items-center py-0.5 border-b border-slate-800">
            <span class="text-slate-400">Ep #${ep.episode}</span>
            <span class="${ep.steps >= 500 ? 'text-emerald-400' : 'text-slate-200'} font-semibold">${ep.steps} steps</span>
            <span class="text-[10px] uppercase text-slate-500">${ep.reason}</span>
          </div>
        `).join('');
      }

      // Pause button label
      el('btn-pause').textContent = data.is_paused ? 'Resume Stream' : 'Pause Stream';
      el('btn-pause').className = data.is_paused 
        ? 'py-2.5 bg-emerald-600 hover:bg-emerald-500 rounded-xl text-xs md:text-sm font-semibold text-white transition-all'
        : 'py-2.5 bg-slate-700 hover:bg-slate-600 rounded-xl text-xs md:text-sm font-semibold text-slate-200 transition-all';
    };

    // Connect Server-Sent Events (SSE)
    const connectSSE = () => {
      const evtSource = new EventSource('/events');
      evtSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          updateTelemetry(data);
        } catch (e) {
          console.error('SSE JSON error', e);
        }
      };
      evtSource.onerror = () => {
        evtSource.close();
        setTimeout(connectSSE, 2000);
      };
    };
    connectSSE();

    // API Post Helper
    const postControl = async (payload) => {
      try {
        const res = await fetch('/api/control', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.telemetry) updateTelemetry(data.telemetry);
      } catch (e) {
        console.error('API Error:', e);
      }
    };

    // Update Model dropdown options filtered by algorithm
    const updateModelSelect = (algo) => {
      const sel = el('select-model');
      sel.innerHTML = '';
      if (algo === 'heuristic' || algo === 'random' || algo === 'manual') {
        sel.disabled = true;
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'No checkpoint required (rule-based / manual)';
        sel.appendChild(opt);
        return;
      }
      sel.disabled = false;
      const list = modelsData[algo] || [];
      const defaultOpt = document.createElement('option');
      defaultOpt.value = '';
      defaultOpt.textContent = `Auto-detected default ${algo.toUpperCase()} checkpoint`;
      sel.appendChild(defaultOpt);

      list.forEach((m) => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m;
        sel.appendChild(opt);
      });
    };

    // Load available models from server and populate dropdown
    const loadModels = async () => {
      try {
        const res = await fetch('/api/models');
        const data = await res.json();
        modelsData = data;
        updateModelSelect(el('select-algo').value);
      } catch (e) {
        console.error('Failed to load models:', e);
      }
    };
    loadModels();

    // Event Listeners for Controls
    el('select-algo').addEventListener('change', (e) => {
      const algo = e.target.value;
      updateModelSelect(algo);
      postControl({ action: 'set_algo', algo: algo, model: el('select-model').value || null });
    });

    el('select-model').addEventListener('change', (e) => {
      postControl({ action: 'set_algo', algo: el('select-algo').value, model: e.target.value || null });
    });

    el('btn-mode-infinite').addEventListener('click', () => {
      el('btn-mode-infinite').className = 'px-3 py-2 rounded-xl text-xs font-semibold bg-sky-500 text-white transition-all';
      el('btn-mode-standard').className = 'px-3 py-2 rounded-xl text-xs font-semibold bg-slate-900 text-slate-400 hover:text-white transition-all';
      postControl({ action: 'set_mode', mode: 'infinite' });
    });

    el('btn-mode-standard').addEventListener('click', () => {
      el('btn-mode-standard').className = 'px-3 py-2 rounded-xl text-xs font-semibold bg-sky-500 text-white transition-all';
      el('btn-mode-infinite').className = 'px-3 py-2 rounded-xl text-xs font-semibold bg-slate-900 text-slate-400 hover:text-white transition-all';
      postControl({ action: 'set_mode', mode: 'standard' });
    });

    el('slider-fps').addEventListener('input', (e) => {
      el('fps-slider-val').textContent = e.target.value;
      postControl({ action: 'set_fps', fps: parseInt(e.target.value, 10) });
    });

    el('slider-sigma').addEventListener('input', (e) => {
      el('sigma-slider-val').textContent = parseFloat(e.target.value).toFixed(2);
      postControl({ action: 'set_sigma', sigma: parseFloat(e.target.value) });
    });

    el('btn-pause').addEventListener('click', () => postControl({ action: 'toggle_pause' }));
    el('btn-reset').addEventListener('click', () => postControl({ action: 'reset' }));

    // Perturbation buttons
    el('btn-push-left').addEventListener('click', () => postControl({ action: 'perturb', type: 'push_left' }));
    el('btn-push-right').addEventListener('click', () => postControl({ action: 'perturb', type: 'push_right' }));
    el('btn-tip-left').addEventListener('click', () => postControl({ action: 'perturb', type: 'tip_left' }));
    el('btn-tip-right').addEventListener('click', () => postControl({ action: 'perturb', type: 'tip_right' }));
    el('btn-shock').addEventListener('click', () => postControl({ action: 'perturb', type: 'shock' }));

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

      if (e.code === 'Space') {
        e.preventDefault();
        postControl({ action: 'toggle_pause' });
      } else if (e.code === 'KeyR') {
        postControl({ action: 'reset' });
      } else if (e.code === 'ArrowLeft') {
        if (el('select-algo').value === 'manual') {
          postControl({ action: 'manual_action', value: 0 });
        } else {
          postControl({ action: 'perturb', type: 'push_left' });
        }
      } else if (e.code === 'ArrowRight') {
        if (el('select-algo').value === 'manual') {
          postControl({ action: 'manual_action', value: 1 });
        } else {
          postControl({ action: 'perturb', type: 'push_right' });
        }
      }
    });
