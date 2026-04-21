import great_expectations as gx
import great_expectations.expectations as gxe


def main():
    print("Initializing Great Expectations workflow...")

    # 1. Creation of a Data Context
    context = gx.get_context()
    print("1. Data Context created.")

    # 2. Creation of a Datasource
    datasource_name = "my_pandas_datasource"
    datasource = context.data_sources.add_pandas(datasource_name)
    print(f"2. Datasource '{datasource_name}' created.")

    # 3. Creation of a Data Asset
    asset_name = "raw_dataset_asset"
    file_path = "data/raw/dataset.csv" 

    try:
        data_asset = datasource.add_csv_asset(
            name=asset_name,
            filepath_or_buffer=file_path
        )
        print(f"3. Data Asset '{asset_name}' created pointing to {file_path}.")
    except Exception as e:
        print(f"Error creating Data Asset (check if file exists): {e}")
        return

    # 4. Creation of a Batch Definition and Batch
    batch_def_name = "full_dataset_batch"
    batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_def_name)
    batch = batch_definition.get_batch()
    print("4. Batch Definition and Batch created successfully.")

    # 5. Definition of Expectations
    suite_name = "basic_validation_suite"
    suite = gx.ExpectationSuite(name=suite_name)
    suite = context.suites.add(suite)

    # Check that required columns exist
    suite.add_expectation(gxe.ExpectColumnToExist(column="text"))
    suite.add_expectation(gxe.ExpectColumnToExist(column="title"))
    suite.add_expectation(gxe.ExpectColumnToExist(column="label"))

    #check format string string int
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(column="text", type_="object")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(column="title", type_="object")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(column="label", type_="int64")
    )
    
    
    # Check that columns do not contain null values
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="text"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="title"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="label"))


    # no special characaters
    regex_no_special_chars = r"^[a-zA-Z0-9\s]+$"

    suite.add_expectation(gxe.ExpectColumnValuesToMatchRegex(
        column="title",
        regex=regex_no_special_chars,
    ))

    suite.add_expectation(gxe.ExpectColumnValuesToMatchRegex(
        column="text",
        regex=regex_no_special_chars,
    ))

    print("5. Expectation Suite created and populated with rules.")

    # 6. Data Batch Validation
    validation_def_name = "dataset_validation"
    validation_definition = gx.ValidationDefinition(
        name=validation_def_name,
        data=batch_definition,
        suite=suite,
    )
    validation_definition = context.validation_definitions.add(validation_definition)

    print("6. Running validation...")

    validation_results = validation_definition.run()

    context.build_data_docs()

    # Open results in browser
    context.open_data_docs()

    # Print summary
    print("\n--- VALIDATION RESULTS ---")
    if validation_results.success:
        print("✅ SUCCESS: Data satisfies all expectations!")
    else:
        print("❌ FAILURE: Data violates one or more expectations.")



if __name__ == "__main__":
    main()