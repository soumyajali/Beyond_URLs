document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const demoBtn = document.getElementById('demoBtn');
    const messageInput = document.getElementById('messageInput');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    
    // Result elements
    const predictionBadge = document.getElementById('predictionBadge');
    const confidenceScore = document.getElementById('confidenceScore');
    const textLength = document.getElementById('textLength');
    const xaiTags = document.getElementById('xaiTags');

    // Demo text for easy testing
    const demoText = "URGENT: Your account at Bank of America has been compromised. We noticed suspicious activity. Please wire $500 to the safe account immediately or your funds will be frozen. Do not contact branch support. http://malicious-link.com";

    demoBtn.addEventListener('click', () => {
        messageInput.value = demoText;
    });

    analyzeBtn.addEventListener('click', async () => {
        const text = messageInput.value.trim();
        
        if (!text) {
            alert("Please enter some text to analyze.");
            return;
        }

        // UI State: Loading
        analyzeBtn.disabled = true;
        results.classList.add('hidden');
        loading.classList.remove('hidden');

        try {
            const response = await fetch('/api/v1/analyze_text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.message || data.error || "Failed to analyze text.");
            }

            // Update UI with results
            renderResults(data);

        } catch (error) {
            // Display error beautifully in the UI instead of a browser alert
            results.classList.remove('hidden');
            predictionBadge.textContent = 'ERROR';
            predictionBadge.className = 'badge deceptive';
            confidenceScore.textContent = '--%';
            textLength.textContent = '0 chars';
            xaiTags.innerHTML = `<p class="xai-desc" style="color: var(--deceptive); font-weight: bold;">
                <i data-feather="alert-triangle"></i> ${error.message}
            </p>`;
            feather.replace(); // re-render icons
        } finally {
            // UI State: Done loading
            loading.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    });

    function renderResults(data) {
        // Show panel
        results.classList.remove('hidden');
        
        // 1. Prediction Badge
        const pred = data.prediction;
        predictionBadge.textContent = pred;
        predictionBadge.className = 'badge ' + pred.split(' ')[0].toLowerCase(); // e.g. "legitimate", "suspicious", "financially"

        // Handle the specific class for deceptive
        if (pred === "Financially Deceptive Phishing") {
            predictionBadge.className = 'badge deceptive';
        }

        // 2. Stats
        confidenceScore.textContent = `${data.confidence.toFixed(2)}%`;
        textLength.textContent = `${data.analyzed_text.length} chars (URLs removed)`;

        // 3. Probabilities
        const probBars = document.getElementById('probBars');
        probBars.innerHTML = '';
        const colorMap = {
            "Legitimate": "var(--legitimate)",
            "Suspicious": "var(--suspicious)",
            "Financially Deceptive Phishing": "var(--deceptive)"
        };
        for (const [cls, prob] of Object.entries(data.probabilities)) {
            probBars.innerHTML += `
                <div class="prob-row">
                    <span class="prob-label">${cls}</span>
                    <div class="prob-bar-container">
                        <div class="prob-bar-fill" style="width: ${prob}%; background-color: ${colorMap[cls]}"></div>
                    </div>
                    <span class="prob-value">${prob}%</span>
                </div>
            `;
        }

        // 4. Indicators
        const indicatorsList = document.getElementById('indicatorsList');
        indicatorsList.innerHTML = '';
        if (data.indicators && data.indicators.length > 0) {
            data.indicators.forEach(ind => {
                const icon = ind.type === 'warning' ? '<i data-feather="alert-triangle"></i>' : '<i data-feather="check-circle"></i>';
                const className = ind.type === 'warning' ? 'indicator-warning' : 'indicator-safe';
                indicatorsList.innerHTML += `<li class="${className}">${icon} ${ind.text}</li>`;
            });
            feather.replace(); // re-render new icons
        }

        // 5. AI Explanation
        const aiExplanation = document.getElementById('aiExplanation');
        aiExplanation.textContent = `"${data.explanation}"`;

        // 6. XAI Tags (SHAP)
        xaiTags.innerHTML = '';
        if (data.xai_explanations && data.xai_explanations.length > 0) {
            data.xai_explanations.forEach(([word, weight]) => {
                const tag = document.createElement('div');
                tag.className = 'xai-tag';
                
                // Color based on weight
                if (weight > 0.05) {
                    tag.style.backgroundColor = 'rgba(239, 68, 68, 0.2)'; 
                    tag.style.border = '1px solid var(--deceptive)';
                    tag.style.color = '#fca5a5';
                } else if (weight < -0.05) {
                    tag.style.backgroundColor = 'rgba(16, 185, 129, 0.2)'; 
                    tag.style.border = '1px solid var(--legitimate)';
                    tag.style.color = '#6ee7b7';
                } else {
                    tag.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
                    tag.style.border = '1px solid var(--border-color)';
                    tag.style.color = 'var(--text-light)';
                }

                tag.innerHTML = `<span>${word}</span> <span style="opacity:0.5; font-size:0.75rem;">(${weight.toFixed(2)})</span>`;
                xaiTags.appendChild(tag);
            });
        } else {
            xaiTags.innerHTML = '<p class="xai-desc">No significant SHAP attributions found.</p>';
        }
    }
});
