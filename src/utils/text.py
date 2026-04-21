import re

def extract_findings_from_report(input_string: str) -> str:
    """Extract findings and subsequent sections from a radiology report.

    Splits the report on uppercase section headers, discards everything
    before ``FINDINGS:`` and removes ``TECHNIQUE:``, ``DOSE:``, and
    ``DLP:`` sections.

    Args:
        input_string: Raw radiology report text.

    Returns:
        Concatenated text of retained sections.
    """
    # Split the input into sections based on titles in capital letters ending with a colon
    sections = re.split(r'(^[A-Z ,_.&]+:\n|(?<=\n)[A-Z ,_.&]+:\n)', input_string)
    
    # Initialize an empty dictionary to store sections
    section_dict = {}
    
    # Iterate through the sections
    for i in range(1, len(sections), 2):  # Using step 2 to pick title and its content
        title = sections[i].strip()  # Get the title (like "FINDINGS:")
        content = sections[i+1] if i + 1 < len(sections) else ""  # Get the content
        section_dict[title] = content.strip()  # Store it in the dictionary
    
    # Remove the "TECHNIQUE:" section if it exists
    if "TECHNIQUE:" in section_dict:
        del section_dict["TECHNIQUE:"]
    if "DOSE:" in section_dict:
        del section_dict["DOSE:"]
    if "DLP:" in section_dict:
        del section_dict["DLP:"]
    
    # If there's a "FINDINGS:" section, remove everything before it
    if "FINDINGS:" in section_dict:
        findings_found = False
        filtered_sections = {}
        
        # Iterate through the sections and only keep those from "FINDINGS:" onward
        for title, content in section_dict.items():
            if title == "FINDINGS:":
                findings_found = True
            if findings_found:
                filtered_sections[title] = content
        
        section_dict = filtered_sections  # Keep sections only after FINDINGS:
    
    # Reconstruct the string by joining titles and their content
    output_string = ""
    for title, content in section_dict.items():
        output_string += f"{title}\n{content}\n\n"  # Add extra newline for spacing
    
    return output_string.strip()  # Remove any extra newlines at the end