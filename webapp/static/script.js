document.getElementById('upload-form').addEventListener('submit', function(e) {
  e.preventDefault();

  const form = e.target;
  const formData = new FormData(form);
  const popup = document.getElementById('processing-popup');
  const timerDisplay = document.getElementById('timer');

  // Show the popup
  popup.style.display = 'block';

  // Start the timer
  let seconds = 0;
  timerDisplay.textContent = '0:00';
  const interval = setInterval(() => {
    seconds++;
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    timerDisplay.textContent = `${minutes}:${secs < 10 ? '0' : ''}${secs}`;
  }, 1000);

  // Send the upload request
  fetch(form.action, {
    method: form.method,
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      clearInterval(interval); // stop the timer
      if (data.redirect) {
        window.location.href = data.redirect;
      }
    })
    .catch(err => {
      clearInterval(interval); // stop the timer on error
      console.error("Upload failed", err);
      alert("Upload failed. Please try again.");
      popup.style.display = 'none';
    });
});
