// JavaScript code for form submission handling
document.getElementById('loanForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the form from submitting via the browser
    // Simulate form submission (replace with actual AJAX call if needed)
    var resultElement = document.getElementById("result");
    var numberOfChildren = document.getElementById("numberOfChildren").value;
    var income = document.getElementById("income").value;
    //var ownCarValue = document.getElementById("ownCar").value;
    var ownCarValue = document.getElementById("ownCar");


    setTimeout(function() {
            // Set the result based on the ownCar value

            if (ownCarValue.checked) {
                document.getElementById('successBanner').style.display = 'block'; // Display the success banner
                successBanner.innerHTML = "Congratulations, your credit card application has been approved!";
            } else {
                document.getElementById('successBanner').style.display = 'block'; // Display the success banner
                successBanner.innerHTML = "Unfortunately, your credit card application was not approved this time.";
            }

    }, 1000); 
});
