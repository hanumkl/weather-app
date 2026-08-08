"""
One-time helper: store your Lakebase URL in a Databricks secret scope.

Run this from a Databricks notebook cell or cluster terminal:
  %sh python setup_secrets.py
  -or-
  python setup_secrets.py

For local dev you can skip this and put LAKEBASE_URL in .env instead.
"""

from __future__ import annotations

import getpass

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import AclPermission


def main() -> None:
    w = WorkspaceClient()
    scope = "database"
    key = "lakebase-url"

    # Create the scope if it doesn't exist
    scopes = {s.name for s in w.secrets.list_scopes()}
    if scope not in scopes:
        print(f"Creating secret scope '{scope}' ...")
        w.secrets.create_scope(scope=scope)
        try:
            me = w.current_user.me().user_name
            w.secrets.put_acl(scope=scope, principal=me, permission=AclPermission.MANAGE)
        except Exception as exc:  # noqa: BLE001
            print(f"(ACL note: {exc})")

    url = getpass.getpass(
        "Paste your Lakebase URL "
        "(postgresql://role:pass@host:5432/databricks_postgres?sslmode=require): "
    )
    if not url.startswith("postgresql://"):
        raise SystemExit("Expected a postgresql:// URL")

    w.secrets.put_secret(scope=scope, key=key, string_value=url)
    print(f"Stored secret {scope}/{key} — lakebase.py and the notebook will pick it up automatically.")


if __name__ == "__main__":
    main()
