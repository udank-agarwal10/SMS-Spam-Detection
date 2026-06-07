/* ============================================
   SMS Spam Detection — Frontend Logic
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {

    // --- DOM References ---
    const messageInput  = document.getElementById('messageInput');
    const charCount     = document.getElementById('charCount');
    const charBadge     = document.getElementById('charBadge');
    const checkButton   = document.getElementById('checkButton');
    const clearButton   = document.getElementById('clearButton');
    const btnText       = checkButton.querySelector('.btn-text');
    const btnLoading    = checkButton.querySelector('.btn-loading');

    const errorToast    = document.getElementById('errorToast');
    const errorMessage  = document.getElementById('errorMessage');

    const resultsWrapper = document.getElementById('resultsWrapper');
    const resultCard     = document.getElementById('resultCard');
    const resultIconWrap = document.getElementById('resultIconWrap');
    const resultIcon     = document.getElementById('resultIcon');
    const resultTitle    = document.getElementById('resultTitle');
    const resultSubtitle = document.getElementById('resultSubtitle');

    const safeBar   = document.getElementById('safeBar');
    const spamBar   = document.getElementById('spamBar');
    const safeValue = document.getElementById('safeValue');
    const spamValue = document.getElementById('spamValue');

    const chartCanvas = document.getElementById('probabilityChart');

    let probabilityChart = null;

    // --- Character Count ---
    messageInput.addEventListener('input', () => {
        const len = messageInput.value.length;
        charCount.textContent = len;

        if (len > 0) {
            charBadge.classList.add('active');
        } else {
            charBadge.classList.remove('active');
        }
    });

    // --- Clear Button ---
    clearButton.addEventListener('click', () => {
        messageInput.value = '';
        charCount.textContent = '0';
        charBadge.classList.remove('active');
        resultsWrapper.classList.add('hidden');
        errorToast.classList.add('hidden');
        messageInput.focus();
    });

    // --- Check / Analyze Button ---
    checkButton.addEventListener('click', async () => {
        const message = messageInput.value.trim();

        // Validation
        if (!message) {
            showError('Please enter a message to analyze.');
            messageInput.focus();
            return;
        }

        // Start loading
        setLoading(true);
        hideError();
        resultsWrapper.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Server error. Please try again.');
            }

            displayResult(data);

        } catch (err) {
            console.error('Prediction error:', err);
            showError(err.message || 'Failed to connect to the server.');
        } finally {
            setLoading(false);
        }
    });

    // --- Keyboard shortcut: Enter to analyze, Shift+Enter for new line ---
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            checkButton.click();
        }
    });

    // --- Display Result ---
    function displayResult(data) {
        resultsWrapper.classList.remove('hidden');

        const isSpam = data.is_spam;
        const spamProb = data.spam_probability;
        const hamProb  = data.ham_probability;

        // Icon & text
        if (isSpam) {
            resultIconWrap.className = 'result-icon-wrap spam';
            resultIcon.textContent = '🚨';
            resultTitle.className = 'spam';
            resultTitle.textContent = 'Spam Detected!';
            resultSubtitle.textContent = 'This message has been flagged as potential spam.';
        } else {
            resultIconWrap.className = 'result-icon-wrap safe';
            resultIcon.textContent = '✅';
            resultTitle.className = 'safe';
            resultTitle.textContent = 'Message is Safe';
            resultSubtitle.textContent = 'This message appears to be legitimate.';
        }

        // Animated probability bars
        requestAnimationFrame(() => {
            safeBar.style.width = hamProb + '%';
            spamBar.style.width = spamProb + '%';
        });

        animateCounter(safeValue, hamProb);
        animateCounter(spamValue, spamProb);

        // Chart
        renderChart(spamProb, hamProb);

        // Scroll to results
        resultsWrapper.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // --- Animate Counter ---
    function animateCounter(element, target) {
        const duration = 1000;
        const start = performance.now();
        const initial = 0;

        function update(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // easeOutQuart
            const eased = 1 - Math.pow(1 - progress, 4);
            const current = initial + (target - initial) * eased;
            element.textContent = current.toFixed(1) + '%';

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }

        requestAnimationFrame(update);
    }

    // --- Render Chart ---
    function renderChart(spamProb, hamProb) {
        if (probabilityChart) {
            probabilityChart.destroy();
        }

        const ctx = chartCanvas.getContext('2d');

        // Gradient fills
        const safeGradient = ctx.createLinearGradient(0, 0, 0, 260);
        safeGradient.addColorStop(0, '#34d399');
        safeGradient.addColorStop(1, '#059669');

        const spamGradient = ctx.createLinearGradient(0, 0, 0, 260);
        spamGradient.addColorStop(0, '#fb7185');
        spamGradient.addColorStop(1, '#e11d48');

        probabilityChart = new Chart(chartCanvas, {
            type: 'doughnut',
            data: {
                labels: ['Spam', 'Safe'],
                datasets: [{
                    data: [spamProb, hamProb],
                    backgroundColor: [spamGradient, safeGradient],
                    borderWidth: 0,
                    hoverOffset: 8,
                    borderRadius: 4,
                    spacing: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '72%',
                layout: {
                    padding: 10
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#94a3b8',
                            padding: 24,
                            usePointStyle: true,
                            pointStyleWidth: 10,
                            font: {
                                family: "'Times New Roman', Times, serif",
                                size: 13,
                                weight: '500'
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(14, 20, 38, 0.95)',
                        titleColor: '#f1f5f9',
                        bodyColor: '#94a3b8',
                        borderColor: 'rgba(99, 102, 241, 0.2)',
                        borderWidth: 1,
                        cornerRadius: 10,
                        padding: 12,
                        bodyFont: {
                            family: "'Inter', sans-serif"
                        },
                        callbacks: {
                            label: function(context) {
                                return ' ' + context.label + ': ' + context.raw.toFixed(1) + '%';
                            }
                        }
                    }
                },
                animation: {
                    animateScale: true,
                    animateRotate: true,
                    duration: 1200,
                    easing: 'easeOutQuart'
                }
            }
        });
    }

    // --- Loading State ---
    function setLoading(isLoading) {
        checkButton.disabled = isLoading;

        if (isLoading) {
            btnText.classList.add('hidden');
            btnLoading.classList.remove('hidden');
        } else {
            btnText.classList.remove('hidden');
            btnLoading.classList.add('hidden');
        }
    }

    // --- Error Handling ---
    function showError(msg) {
        errorMessage.textContent = msg;
        errorToast.classList.remove('hidden');
    }

    function hideError() {
        errorToast.classList.add('hidden');
    }

});
