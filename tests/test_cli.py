import json
from pathlib import Path
from llm_batch import cli
import re

preamble = "Option [A-D]. Speaking only of the subjects who took part in this particular study,"

version2 = r"In considering this question, it might be helpful to bear in mind the six principles articulated in the 2016 American Statistical Association \*Statement on Statistical Significance and \*P\*\-values\*\:"

version3 = r"In considering this question, it might be helpful to bear in mind a principle articulated in the 2016 American Statistical Association"


def count_occurrences(pattern, text):
    return len(re.findall(pattern, text))


def test_config(capsys):
    cli.config()
    captured = capsys.readouterr()
    assert "logging" in captured.out or "logging" in captured.err


def test_prompt_test_mode(tmp_path, capsys):
    # Create a fake template file
    template_content = '{"model": "gpt-3", "messages": [{"role": "user", "content": "{{ question }}"}]}'
    template_file = tmp_path / "template.jinja2"
    template_file.write_text(template_content)

    # Create a fake data file
    data_content = (
        'questions:\n  - question: "What is AI?"\n  - question: "What is ML?"'
    )
    data_file = tmp_path / "data.yaml"
    data_file.write_text(data_content)

    # Output directory
    out_dir = tmp_path / "output"

    # Patch extract_combinations to yield test data
    class DummyF:
        @staticmethod
        def extract_combinations(yaml_data):
            return [{"question": "What is AI?"}, {"question": "What is ML?"}]

    extract_combinations = cli.extract_combinations
    cli.extract_combinations = DummyF.extract_combinations
    try:
        cli.template(
            template=template_file, data=data_file, out=out_dir, execute=False
        )
        captured = capsys.readouterr()
        assert "Executed combination" in captured.out
        assert out_dir.exists()
    finally:
        cli.extract_combinations = extract_combinations  # Restore original function


def test_dichotomania_q1_output():
    # get path of current file
    current_file_path = Path(__file__).resolve()
    q1_output_path = current_file_path.parent / "dichotomania" / "q1-output"
    model_output_dirs = [d for d in q1_output_path.iterdir() if d.is_dir()]
    for output_dir in model_output_dirs:
        output_files = list(output_dir.glob("*.json"))
        assert len(output_files) == 30
        for file in output_files:
            data = json.load(open(file, "r"))
            assert isinstance(data, dict)
            assert "template_params" in data
            assert "request" in data
            assert "response" in data
            assert data["template_params"]["model"] == output_dir.name.replace("_", "/")

            # make sure the template params resulted in the proper prompt text generation
            text = data["request"]["messages"][0]["content"][0]["text"]
            if data["template_params"]["preamble"] == "yes":
                assert count_occurrences(preamble, text) == 4
            else:
                assert count_occurrences(preamble, text) == 0

            if data["template_params"]["version"] == "v2":
                assert count_occurrences(version2, text) == 1
            elif data["template_params"]["version"] == "v3":
                assert count_occurrences(version3, text) == 1
            else:
                assert count_occurrences(version2, text) == 0
                assert count_occurrences(version3, text) == 0

            pvalue = data["template_params"]["pvalue"]
            assert count_occurrences(rf"\*p\* = {pvalue}", text) == 1


def test_dichotomania_q2_output():
    # get path of current file
    current_file_path = Path(__file__).resolve()
    q1_output_path = current_file_path.parent / "dichotomania" / "q2-output"
    model_output_dirs = [d for d in q1_output_path.iterdir() if d.is_dir()]
    for output_dir in model_output_dirs:
        output_files = list(output_dir.glob("*.json"))
        assert len(output_files) == 15
        for file in output_files:
            data = json.load(open(file, "r"))
            assert isinstance(data, dict)
            assert "template_params" in data
            assert "request" in data
            assert "response" in data
            assert data["template_params"]["model"] == output_dir.name.replace("_", "/")

            # make sure the template params resulted in the proper prompt text generation
            text = data["request"]["messages"][0]["content"][0]["text"]

            if data["template_params"]["version"] == "v2":
                assert count_occurrences(version2, text) == 1
            elif data["template_params"]["version"] == "v3":
                assert count_occurrences(version3, text) == 1
            else:
                assert count_occurrences(version2, text) == 0
                assert count_occurrences(version3, text) == 0

            pvalue = data["template_params"]["pvalue"]
            assert count_occurrences(rf"\*p\*-value of {pvalue}", text) == 1


def test_dichotomania_q3_output():
    # get path of current file
    current_file_path = Path(__file__).resolve()
    q1_output_path = current_file_path.parent / "dichotomania" / "q3-output"
    model_output_dirs = [d for d in q1_output_path.iterdir() if d.is_dir()]
    for output_dir in model_output_dirs:
        output_files = list(output_dir.glob("*.json"))
        assert len(output_files) == 15
        for file in output_files:
            data = json.load(open(file, "r"))
            assert isinstance(data, dict)
            assert "template_params" in data
            assert "request" in data
            assert "response" in data
            assert data["template_params"]["model"] == output_dir.name.replace("_", "/")

            # make sure the template params resulted in the proper prompt text generation
            text = data["request"]["messages"][0]["content"][0]["text"]

            if data["template_params"]["version"] == "v2":
                assert count_occurrences(version2, text) == 1
            elif data["template_params"]["version"] == "v3":
                assert count_occurrences(version3, text) == 1
            else:
                assert count_occurrences(version2, text) == 0
                assert count_occurrences(version3, text) == 0

            pvalue = data["template_params"]["pvalue"]
            assert count_occurrences(rf"\*p\*-value of {pvalue}", text) == 1
