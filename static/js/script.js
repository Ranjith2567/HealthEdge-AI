document.addEventListener('DOMContentLoaded', function() {
    const probInput = document.getElementById('js_prob');
    if (probInput) {
        const ctx = document.getElementById('riskChart').getContext('2d');
        const riskVal = parseFloat(probInput.value.replace('%', ''));
        
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Risk', 'Safe'],
                datasets: [{
                    data: [riskVal, 100 - riskVal],
                    backgroundColor: ['#ef4444', '#10b981'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '75%',
                plugins: { legend: { display: false } }
            }
        });
    }
});