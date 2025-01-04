<?php
include 'db.php'; // Include the database connection script

if (isset($_GET['character'])) {
    $character = $_GET['character'];
    $sql = "INSERT INTO recognition_results (character_predicted) VALUES ('$character')";
    if ($conn->query($sql) === TRUE) {
        echo "Result saved to the database";
    } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
    }
} else {
    echo "No character provided for insertion.";
}

$conn->close();
