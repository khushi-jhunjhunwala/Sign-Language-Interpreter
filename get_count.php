<?php
// Connect to the database
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "signlang";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Get the character ('y' or 'n') from the URL query parameter
$character = "$_GET['character']";

// Query the database to count the occurrences of the character
$sql = "SELECT COUNT(*) AS count FROM recognition_results WHERE character_predicted = '$character'";
$result = $conn->query($sql);

if ($result) {
    $row = $result->fetch_assoc();
    echo $row['count'];
} else {
    echo "Error: " . $conn->error;
}

$conn->close();
?>
