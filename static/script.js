// script.js — AlzheAI
const API_BASE_URL = 'http://127.0.0.1:5000';

// ====================================================================
// TAB NAVIGATION
// ====================================================================
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function () {
        const targetTab = this.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(targetTab).classList.add('active');
    });
});

// ====================================================================
// HELPERS
// ====================================================================

/**
 * Animasikan probability bar setelah elemen ada di DOM.
 * Sedikit delay agar CSS transition bisa berjalan.
 */
function animateBars() {
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            document.querySelectorAll('.probability-bar[data-width]').forEach(bar => {
                bar.style.width = bar.dataset.width + '%';
            });
        });
    });
}

/**
 * Setup file input + drag-drop untuk sebuah upload area.
 * @param {object} ids - { area, fileInput, preview, placeholder, btnId }
 * @param {function} onFile - callback(file)
 */
function setupUploadArea({ area, fileInput, preview, placeholder }) {
    area.addEventListener('click', () => fileInput.click());

    area.addEventListener('dragover', e => {
        e.preventDefault();
        area.classList.add('drag-over');
    });

    area.addEventListener('dragleave', () => {
        area.classList.remove('drag-over');
    });

    area.addEventListener('drop', e => {
        e.preventDefault();
        area.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            loadPreview(file, preview, placeholder, area);
            fileInput._droppedFile = file;
        }
    });

    fileInput.addEventListener('change', e => {
        if (e.target.files[0]) {
            loadPreview(e.target.files[0], preview, placeholder, area);
        }
    });
}

function loadPreview(file, previewEl, placeholderEl, areaEl) {
    const reader = new FileReader();
    reader.onload = e => {
        previewEl.src = e.target.result;
        previewEl.style.display = 'block';
        placeholderEl.style.display = 'none';
        areaEl.classList.add('has-image');
    };
    reader.readAsDataURL(file);
}

function getFile(fileInput) {
    return fileInput._droppedFile || fileInput.files[0] || null;
}

// ====================================================================
// KLASIFIKASI TAB
// ====================================================================

const uploadAreaKlasifikasi   = document.getElementById('uploadAreaKlasifikasi');
const fileInputKlasifikasi    = document.getElementById('fileInputKlasifikasi');
const previewKlasifikasi      = document.getElementById('previewKlasifikasi');
const placeholderKlasifikasi  = document.getElementById('uploadPlaceholderKlasifikasi');
const btnKlasifikasi          = document.getElementById('btnKlasifikasi');
const resultKlasifikasi       = document.getElementById('resultKlasifikasi');

setupUploadArea({
    area: uploadAreaKlasifikasi,
    fileInput: fileInputKlasifikasi,
    preview: previewKlasifikasi,
    placeholder: placeholderKlasifikasi,
});

btnKlasifikasi.addEventListener('click', async () => {
    const file = getFile(fileInputKlasifikasi);
    if (!file) {
        resultKlasifikasi.className = 'result-box';
        resultKlasifikasi.innerHTML = '<p class="message message-error">Mohon upload gambar terlebih dahulu.</p>';
        return;
    }

    btnKlasifikasi.disabled = true;
    btnKlasifikasi.innerHTML = '<span class="loading"></span>Menganalisis…';
    resultKlasifikasi.className = 'result-box';
    resultKlasifikasi.innerHTML = '<p class="message message-info">Sedang menganalisis gambar MRI…</p>';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (response.ok) {
            const sorted = Object.entries(result.probabilities).sort(([, a], [, b]) => b - a);

            let barsHtml = sorted.map(([key, val]) => {
                const pct = (val * 100).toFixed(2);
                return `
                    <div class="probability-item">
                        <div class="probability-header">
                            <span class="probability-name">${key}</span>
                            <span class="probability-value">${pct}%</span>
                        </div>
                        <div class="probability-bar-bg">
                            <div class="probability-bar" data-width="${pct}" style="width:0%"></div>
                        </div>
                    </div>`;
            }).join('');

            resultKlasifikasi.innerHTML = `
                <div class="result-header">
                    <div class="result-icon">✓</div>
                    <span class="result-title">Hasil Prediksi</span>
                </div>
                <div class="prediction-main">
                    <div class="prediction-label">Diagnosis</div>
                    <div class="prediction-value">${result.prediction}</div>
                    <div class="prediction-confidence">${result.confidence.toFixed(2)}% confidence</div>
                </div>
                <div class="probability-section">${barsHtml}</div>
            `;

            animateBars();
        } else {
            resultKlasifikasi.innerHTML = `<p class="message message-error">Error: ${result.error || 'Terjadi kesalahan.'}</p>`;
        }
    } catch {
        resultKlasifikasi.innerHTML = `<p class="message message-error">Gagal terhubung ke server. Pastikan Flask berjalan di ${API_BASE_URL}.</p>`;
    } finally {
        btnKlasifikasi.disabled = false;
        btnKlasifikasi.textContent = 'Klasifikasi';
    }
});

// ====================================================================
// ENHANCE TAB
// ====================================================================

const uploadAreaEnhance  = document.getElementById('uploadAreaEnhance');
const fileInputEnhance   = document.getElementById('fileInputEnhance');
const previewEnhance     = document.getElementById('previewEnhance');
const placeholderEnhance = document.getElementById('uploadPlaceholderEnhance');
const btnEnhance         = document.getElementById('btnEnhance');
const resultEnhance      = document.getElementById('resultEnhance');

setupUploadArea({
    area: uploadAreaEnhance,
    fileInput: fileInputEnhance,
    preview: previewEnhance,
    placeholder: placeholderEnhance,
});

let enhancedImageData = null;

btnEnhance.addEventListener('click', async () => {
    const file = getFile(fileInputEnhance);
    if (!file) {
        resultEnhance.className = 'result-box';
        resultEnhance.innerHTML = '<p class="message message-error">Mohon upload gambar terlebih dahulu.</p>';
        return;
    }

    btnEnhance.disabled = true;
    btnEnhance.innerHTML = '<span class="loading"></span>Memproses…';
    resultEnhance.className = 'result-box';
    resultEnhance.innerHTML = '<p class="message message-info">Sedang melakukan enhancement gambar…</p>';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/enhance`, {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (response.ok && result.success) {
            enhancedImageData = result.enhanced_image_base64;

            resultEnhance.innerHTML = `
                <div class="result-header">
                    <div class="result-icon">✨</div>
                    <span class="result-title">Enhancement Selesai</span>
                </div>
                <div class="upload-area has-image" style="cursor:default; margin-bottom: 0;">
                    <img src="data:image/jpeg;base64,${result.enhanced_image_base64}"
                         class="preview-image" alt="Enhanced MRI">
                </div>
                <div style="text-align:center;">
                    <button class="download-btn" onclick="downloadEnhancedImage()">
                        ↓ Unduh Gambar
                    </button>
                </div>
            `;
        } else {
            resultEnhance.innerHTML = `<p class="message message-error">Error: ${result.error || 'Terjadi kesalahan.'}</p>`;
        }
    } catch {
        resultEnhance.innerHTML = `<p class="message message-error">Gagal terhubung ke server. Pastikan Flask berjalan di ${API_BASE_URL}.</p>`;
    } finally {
        btnEnhance.disabled = false;
        btnEnhance.textContent = 'Enhance';
    }
});

// ====================================================================
// DOWNLOAD
// ====================================================================

window.downloadEnhancedImage = function () {
    if (enhancedImageData) {
        const link = document.createElement('a');
        link.href = `data:image/jpeg;base64,${enhancedImageData}`;
        link.download = 'enhanced_mri.jpg';
        link.click();
    }
};