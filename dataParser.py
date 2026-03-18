import pandas as pd

def parse_data(file_path):
    ### Values are in the format:id,url,region,region_url,price,year,manufacturer,model,condition,cylinders,fuel,odometer,title_status,transmission,VIN,drive,size,type,paint_color,image_url,description,county,state,lat,long,posting_date
    # More readable format:
    # ID, url, region, region_url, price, year, 
    # manufacturer, model, condition, cylinders, fuel, 
    # odometer, title_status, transmission, VIN, drive, size, 
    # type, paint_color, image_url, description, 
    # county, state, lat, long, posting_date
    ### We only care about the following:
    # ID, Price, Year, Manufacturer, Model, 
    # Condition, Cylinders, Fuel, Odometer, 
    # Title_status, Transmission, Drive, Type, Paint Color

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    # Select only the relevant columns
    relevant_columns = ['id', 'price', 'year', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'type', 'paint_color']
    df = df[relevant_columns]
    # Create new file using selected values
    df.to_csv('parsed_data.csv', index=False)
    # Save the data and safely export it to a new file
    print("Data parsed and saved to 'parsed_data.csv'.")

if __name__ == "__main__":
    parse_data('data.csv')