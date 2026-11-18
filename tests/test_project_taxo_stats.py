from concurrent.futures import ThreadPoolExecutor

from API_operations.helpers.Service import Service
from BO.Project import ProjectBO
from DB.Project import ProjectTaxoStat


def test_update_taxo_stats_concurrent(database):
    def run_update():
        with Service() as sce:
            ProjectBO.update_taxo_stats(sce.session, 1)
            sce.session.commit()

    # Run initial update so rows exist in projects_taxo_stat
    run_update()

    # Run concurrent updates on the same project; without row-level locking
    # on projects, concurrent DELETE + INSERT races with UniqueViolation.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for _ in range(5):
            futures = [pool.submit(run_update), pool.submit(run_update)]
            for fut in futures:
                fut.result()

    with Service() as sce:
        stats = (
            sce.session.query(ProjectTaxoStat)
            .filter(ProjectTaxoStat.projid == 1)
            .all()
        )
        assert len(stats) > 0
