// Basic JavaScript for Form Validation and UI updates
document.addEventListener('DOMContentLoaded', function() {
    // Form validation
    const preferenceForm = document.getElementById('preferenceForm');
    if (preferenceForm) {
        preferenceForm.addEventListener('submit', function(e) {
            const interests = document.getElementById('interests').value;
            if (!interests.trim()) {
                e.preventDefault();
                alert('Please enter your interests to get recommendations.');
            }
        });
    }

    // Auto-hide alerts after 3 seconds
    const alerts = document.querySelectorAll('.alert');
    if (alerts.length > 0) {
        setTimeout(() => {
            alerts.forEach(alert => {
                alert.style.display = 'none';
            });
        }, 3000);
    }
});
