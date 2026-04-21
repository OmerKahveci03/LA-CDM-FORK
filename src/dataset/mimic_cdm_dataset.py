from torch.utils.data import Dataset
import pandas as pd
import ast

from src.utils.text import extract_findings_from_report

class MIMICCDMDataset(Dataset):
    """MIMIC-CDM dataset for clinical decision making.

    Pre-processes patient records by resolving lab test IDs to human-readable
    names and extracting radiology report findings, producing a flat table
    with one column per test.
    """

    def __init__(
        self,
        data_file: str,
        lab_test_mapping_file: str,
        lab_tests: list[str],
        imaging_tests: list[str],
        other_tests: list[str],
        small_sample: bool = False,
    ) -> None:
        self.data = pd.read_csv(data_file)
        self.lab_test_mapping = pd.read_csv(lab_test_mapping_file)
        self.all_tests = lab_tests + imaging_tests + other_tests
        for test in lab_tests + imaging_tests:
            self.data[test] = ""
        self.test_ids = {}
        self.id_to_name = {}
        for test in lab_tests:
            corresponding_ids = ast.literal_eval(self.lab_test_mapping[self.lab_test_mapping['label'] == test]['corresponding_ids'].item())
            self.test_ids[test] = corresponding_ids
            for id in corresponding_ids:
                self.id_to_name[id] = self.lab_test_mapping[self.lab_test_mapping['itemid'] == ast.literal_eval(self.lab_test_mapping[self.lab_test_mapping['itemid'] == id]['corresponding_ids'].item())[0]]['label'].item()
        # Iterate over data and populate the test columns with the corresponding values
        for idx, row in self.data.iterrows():
            lab_results = row['Laboratory Tests']
            lab_results = ast.literal_eval(lab_results)
            for test in lab_tests:
                test_string = ""
                for id in self.test_ids[test]:
                    if id in lab_results:
                        test_string += self.id_to_name[id] + ": " + lab_results[id] + ", "
                if test_string != "":
                    self.data.at[idx, test] = test_string[:-2] + "\n"
                else:
                    self.data.at[idx, test] = "not available.\n"
            imaging_results = row['Radiology']
            imaging_results = ast.literal_eval(imaging_results)
            for test in imaging_tests:
                modality_list = []
                for result in imaging_results:
                    if result.get('Modality') == test:
                        modality_list.append(extract_findings_from_report(result.get('Report')))
                if len(modality_list) == 0:
                    self.data.at[idx, test] = "not available.\n"
                elif len(modality_list) == 1:
                    self.data.at[idx, test] = modality_list[0]
                else:
                    report_string = "Report 1:\n"
                    for i, report in enumerate(modality_list):
                        report_string += report
                        if i != len(modality_list) - 1:
                            report_string += f'\nReport {i+2}:\n'
                    self.data.at[idx, test] = f"There are {len(modality_list)} reports available for this test:\n{report_string}"
        columns_to_keep = ['Patient ID', 'Patient History Summary'] + self.all_tests + ['Label']
        self.data = self.data[columns_to_keep]
        if small_sample:
            self.data = self.data.iloc[:2]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, str | dict[str, str]]:
        item = self.data.iloc[idx]
        test_results = item[self.all_tests].to_dict()
        item = item[['Patient ID', 'Patient History Summary', 'Label']].to_dict()
        item['Test Results'] = test_results

        return {
            'patient_id': item['Patient ID'],
            'patient_history': item['Patient History Summary'],
            'condition': item['Label'],
            'test_results': item['Test Results']
        }
        