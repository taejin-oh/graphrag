import pandas as pd

from graphrag.query.indexer_adapters import read_indexer_reports


def _community_reports_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "r-level-2",
                "community": 10,
                "level": 2,
                "title": "Community L2",
                "summary": "s2",
                "full_content": "c2",
                "rank": 1,
                "findings": [],
            },
            {
                "id": "r-level-3",
                "community": 20,
                "level": 3,
                "title": "Community L3",
                "summary": "s3",
                "full_content": "c3",
                "rank": 1,
                "findings": [],
            },
        ]
    )


def _communities_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "c-level-2",
                "community": 10,
                "level": 2,
                "title": "Community L2",
                "entity_ids": ["e1", "e2"],
            },
            {
                "id": "c-level-3",
                "community": 20,
                "level": 3,
                "title": "Community L3",
                "entity_ids": ["e1", "e2"],
            },
        ]
    )


def test_read_indexer_reports_rolls_up_to_max_level_per_entity():
    reports = read_indexer_reports(
        final_community_reports=_community_reports_df(),
        final_communities=_communities_df(),
        community_level=3,
        dynamic_community_selection=False,
    )

    assert [report.short_id for report in reports] == ["20"]


def test_read_indexer_reports_honors_lower_community_level_cutoff():
    reports = read_indexer_reports(
        final_community_reports=_community_reports_df(),
        final_communities=_communities_df(),
        community_level=2,
        dynamic_community_selection=False,
    )

    assert [report.short_id for report in reports] == ["10"]
