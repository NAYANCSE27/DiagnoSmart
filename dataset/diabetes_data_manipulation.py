import csv

# Open the input CSV file
with open('input_file.csv', 'r') as input_file:
    # Create a CSV reader object
    reader = csv.reader(input_file)

    # Open the output CSV file
    with open('output_file.csv', 'w', newline='') as output_file:
        # Create a CSV writer object
        writer = csv.writer(output_file)

        # Loop through each row in the input CSV file
        for row in reader:
            # Do something with the row (e.g., print it)
            print(row)

            # Add an extra value to the end of the row
            row.append('Extra Value')

            # Write the modified row to the output CSV file
            writer.writerow(row)
