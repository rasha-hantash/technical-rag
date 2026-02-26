"""Scoring engine for PDF extraction evaluation.

Compares parser output against ground truth annotations to compute
precision, recall, accuracy, and text similarity metrics.
"""

import difflib
import json
from dataclasses import dataclass, field

from .ground_truth import GroundTruth, PageAnnotation, Verdict

# Imports needed only for type hints from parser models
from pdf_llm_server.rag.ingestion.parser_models import ParsedDocument, ParsedPage

# Minimum SequenceMatcher ratio to consider a cross-parser block match
_MATCH_THRESHOLD = 0.6


@dataclass
class PageScore:
    """Scoring results for a single page."""

    page_number: int
    correct: int = 0
    partial: int = 0
    wrong: int = 0
    missing: int = 0
    total_extracted: int = 0
    matched: int = 0
    text_similarities: list[float] = field(default_factory=list)


@dataclass
class ScoreReport:
    """Aggregate scoring results across all pages."""

    parser_name: str
    gt_parser_name: str
    same_parser: bool
    page_scores: list[PageScore] = field(default_factory=list)
    # Overall metrics computed by finalize()
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    avg_text_similarity: float = 0.0
    missing_rate: float = 0.0
    total_annotated: int = 0
    total_extracted: int = 0
    total_missing: int = 0


def align_blocks_same_parser(
    gt_page: PageAnnotation, parsed_page: ParsedPage
) -> PageScore:
    """Align blocks by index when the scored parser matches the ground truth parser.

    Since annotations were made against the same parser's output, block indices
    map directly between ground truth and parsed output.
    """
    score = PageScore(page_number=gt_page.page_number)
    score.total_extracted = len(parsed_page.blocks)

    # Build index lookup for parsed blocks
    blocks_by_index = {b.block_index: b for b in parsed_page.blocks}

    for bv in gt_page.block_verdicts:
        block = blocks_by_index.get(bv.block_index)
        if block is None:
            # Block was annotated but no longer exists in output
            score.wrong += 1
            continue

        score.matched += 1

        if bv.verdict == Verdict.correct:
            score.correct += 1
        elif bv.verdict == Verdict.partial:
            score.partial += 1
            if bv.corrected_text:
                ratio = difflib.SequenceMatcher(
                    None, bv.original_text, bv.corrected_text
                ).ratio()
                score.text_similarities.append(ratio)
        else:
            score.wrong += 1
            if bv.corrected_text:
                ratio = difflib.SequenceMatcher(
                    None, bv.original_text, bv.corrected_text
                ).ratio()
                score.text_similarities.append(ratio)

    score.missing = len(gt_page.missing_blocks)
    return score


def align_blocks_cross_parser(
    gt_page: PageAnnotation, parsed_page: ParsedPage
) -> PageScore:
    """Align blocks via fuzzy text matching when parsers differ.

    Uses greedy best-match-first strategy: for each ground truth verdict,
    find the most similar extracted block above the match threshold.
    """
    score = PageScore(page_number=gt_page.page_number)
    score.total_extracted = len(parsed_page.blocks)

    # Candidate pool — blocks that haven't been matched yet
    candidates = {b.block_index: b for b in parsed_page.blocks}

    for bv in gt_page.block_verdicts:
        best_ratio = 0.0
        best_index = None

        for idx, block in candidates.items():
            ratio = difflib.SequenceMatcher(
                None, bv.original_text, block.text
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_index = idx

        if best_index is not None and best_ratio >= _MATCH_THRESHOLD:
            matched_block = candidates.pop(best_index)
            score.matched += 1

            if bv.verdict == Verdict.correct:
                score.correct += 1
            elif bv.verdict == Verdict.partial:
                score.partial += 1
                if bv.corrected_text:
                    sim = difflib.SequenceMatcher(
                        None, matched_block.text, bv.corrected_text
                    ).ratio()
                    score.text_similarities.append(sim)
            else:
                score.wrong += 1
                if bv.corrected_text:
                    sim = difflib.SequenceMatcher(
                        None, matched_block.text, bv.corrected_text
                    ).ratio()
                    score.text_similarities.append(sim)
        else:
            # Could not find a matching block for this verdict
            score.wrong += 1

    score.missing = len(gt_page.missing_blocks)
    return score


def finalize(report: ScoreReport) -> None:
    """Compute aggregate metrics from per-page scores. Mutates report in place."""
    total_correct = sum(ps.correct for ps in report.page_scores)
    total_partial = sum(ps.partial for ps in report.page_scores)
    total_wrong = sum(ps.wrong for ps in report.page_scores)
    total_matched = sum(ps.matched for ps in report.page_scores)
    total_missing = sum(ps.missing for ps in report.page_scores)
    total_extracted = sum(ps.total_extracted for ps in report.page_scores)

    total_annotated = total_correct + total_partial + total_wrong

    report.total_annotated = total_annotated
    report.total_extracted = total_extracted
    report.total_missing = total_missing

    if total_annotated > 0:
        report.accuracy = (total_correct + total_partial * 0.5) / total_annotated
    else:
        report.accuracy = 0.0

    if total_extracted > 0:
        report.precision = total_matched / total_extracted
    else:
        report.precision = 0.0

    denominator = total_annotated + total_missing
    if denominator > 0:
        report.recall = total_matched / denominator
        report.missing_rate = total_missing / denominator
    else:
        report.recall = 0.0
        report.missing_rate = 0.0

    all_similarities = []
    for ps in report.page_scores:
        all_similarities.extend(ps.text_similarities)
    if all_similarities:
        report.avg_text_similarity = sum(all_similarities) / len(all_similarities)
    else:
        report.avg_text_similarity = 0.0


def score_extraction(
    gt: GroundTruth, doc: ParsedDocument, parser_name: str
) -> ScoreReport:
    """Score a parsed document against ground truth annotations.

    Args:
        gt: Ground truth annotations.
        doc: Parsed document output from the parser being evaluated.
        parser_name: Name of the parser that produced doc (e.g. "pymupdf").

    Returns:
        ScoreReport with per-page and aggregate metrics.
    """
    same_parser = gt.parser_name == parser_name
    report = ScoreReport(
        parser_name=parser_name,
        gt_parser_name=gt.parser_name,
        same_parser=same_parser,
    )

    # Build page lookup for parsed document
    parsed_pages_by_num = {p.page_number: p for p in doc.pages}

    align_fn = align_blocks_same_parser if same_parser else align_blocks_cross_parser

    for gt_page in gt.pages:
        parsed_page = parsed_pages_by_num.get(gt_page.page_number)
        if parsed_page is None:
            # Page exists in ground truth but not in parsed output
            annotated_count = len(gt_page.block_verdicts)
            missing_count = len(gt_page.missing_blocks)
            report.page_scores.append(
                PageScore(
                    page_number=gt_page.page_number,
                    wrong=annotated_count,
                    missing=missing_count,
                )
            )
            continue

        page_score = align_fn(gt_page, parsed_page)
        report.page_scores.append(page_score)

    finalize(report)
    return report


def format_report(report: ScoreReport) -> str:
    """Format a ScoreReport as a human-readable text table."""
    same_label = "same parser" if report.same_parser else "cross parser"
    lines = [
        "Extraction Score Report",
        "=======================",
        f"Parser scored:  {report.parser_name}",
        f"Ground truth:   {report.gt_parser_name} ({same_label})",
        "",
        "Per-page breakdown:",
        "Page | Correct | Partial | Wrong | Missing | Extracted | Matched | Accuracy",
        "---- | ------- | ------- | ----- | ------- | --------- | ------- | --------",
    ]

    for ps in report.page_scores:
        annotated = ps.correct + ps.partial + ps.wrong
        if annotated > 0:
            page_acc = (ps.correct + ps.partial * 0.5) / annotated * 100
        else:
            page_acc = 0.0
        lines.append(
            f"{ps.page_number:4d} | {ps.correct:7d} | {ps.partial:7d} | "
            f"{ps.wrong:5d} | {ps.missing:7d} | {ps.total_extracted:9d} | "
            f"{ps.matched:7d} | {page_acc:6.1f}%"
        )

    lines.extend(
        [
            "",
            "Overall Metrics:",
            f"  Accuracy:        {report.accuracy * 100:5.1f}%",
            f"  Precision:       {report.precision * 100:5.1f}%",
            f"  Recall:          {report.recall * 100:5.1f}%",
            f"  Text Similarity: {report.avg_text_similarity * 100:5.1f}%",
            f"  Missing Rate:    {report.missing_rate * 100:5.1f}%",
            f"  Total Annotated: {report.total_annotated:5d}",
            f"  Total Extracted: {report.total_extracted:5d}",
            f"  Total Missing:   {report.total_missing:5d}",
        ]
    )

    return "\n".join(lines)


def format_report_json(report: ScoreReport) -> str:
    """Format a ScoreReport as a JSON string."""
    page_data = []
    for ps in report.page_scores:
        annotated = ps.correct + ps.partial + ps.wrong
        if annotated > 0:
            page_acc = (ps.correct + ps.partial * 0.5) / annotated
        else:
            page_acc = 0.0
        page_data.append(
            {
                "page_number": ps.page_number,
                "correct": ps.correct,
                "partial": ps.partial,
                "wrong": ps.wrong,
                "missing": ps.missing,
                "total_extracted": ps.total_extracted,
                "matched": ps.matched,
                "accuracy": round(page_acc, 4),
            }
        )

    data = {
        "parser_name": report.parser_name,
        "gt_parser_name": report.gt_parser_name,
        "same_parser": report.same_parser,
        "pages": page_data,
        "overall": {
            "accuracy": round(report.accuracy, 4),
            "precision": round(report.precision, 4),
            "recall": round(report.recall, 4),
            "avg_text_similarity": round(report.avg_text_similarity, 4),
            "missing_rate": round(report.missing_rate, 4),
            "total_annotated": report.total_annotated,
            "total_extracted": report.total_extracted,
            "total_missing": report.total_missing,
        },
    }
    return json.dumps(data, indent=2)
