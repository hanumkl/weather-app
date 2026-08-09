# Databricks notebook source
# MAGIC %md
# MAGIC # Which Lakebase Instance Should I Use?
# MAGIC
# MAGIC Run this to find the Lakebase instance this project should point at, then
# MAGIC put its **name** in:
# MAGIC
# MAGIC - the `lakebase_instance` widget of `ingest_weather_embeddings`, and
# MAGIC - `LAKEBASE_INSTANCE_NAME` in `app.yaml` (then redeploy the app).
# MAGIC
# MAGIC Both must match, or the app searches a different database than the one the
# MAGIC notebook wrote to.
# MAGIC
# MAGIC This connects with your own Databricks identity using a short-lived token,
# MAGIC so no stored password is involved.

# COMMAND ----------

# MAGIC %pip uninstall -y psycopg2 psycopg2-binary

# COMMAND ----------

# MAGIC %pip install -q 'databricks-sdk>=0.61.0'

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,List every Lakebase instance you can see
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
me = w.current_user.me().user_name
print(f"Your Databricks identity (= your Postgres role): {me}\n")

instances = list(w.database.list_database_instances())
if not instances:
    print("No Lakebase instances visible. Create one in Compute > Database instances.")
for inst in instances:
    print(f"  name={inst.name}")
    print(f"    state={inst.state}")
    print(f"    host={inst.read_write_dns}")

# COMMAND ----------

# DBTITLE 1,Check each instance for weather tables
import uuid

import psycopg2

DATABASE = "databricks_postgres"


def inspect(instance_name):
    """Connect with a freshly minted token and report the weather tables."""
    inst = w.database.get_database_instance(name=instance_name)
    cred = w.database.generate_database_credential(
        request_id=str(uuid.uuid4()), instance_names=[instance_name]
    )
    conn = psycopg2.connect(
        host=inst.read_write_dns,
        port=5432,
        dbname=DATABASE,
        user=me,
        password=cred.token,
        sslmode="require",
        connect_timeout=15,
    )
    try:
        with conn.cursor() as cur:
            counts = {}
            for table in ("weather_documents", "weather_embeddings"):
                cur.execute("SELECT to_regclass(%s)", (table,))
                if cur.fetchone()[0] is None:
                    counts[table] = None
                    continue
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cur.fetchone()[0]
            return counts
    finally:
        conn.close()


for inst in instances:
    print(f"\n{inst.name}")
    try:
        for table, count in inspect(inst.name).items():
            print(f"  {table}: {'MISSING' if count is None else f'{count} rows'}")
    except Exception as exc:  # noqa: BLE001 - surface whatever blocks the connection
        print(f"  cannot inspect: {type(exc).__name__}: {exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Picking one
# MAGIC
# MAGIC - An instance already holding `weather_documents` rows is the one your
# MAGIC   earlier sync wrote to — point both the app and notebook there.
# MAGIC - If every instance shows `MISSING`, pick the instance you want to use, set
# MAGIC   it in both places, and run `POST /weather/sync` to populate it. The
# MAGIC   notebook creates the tables automatically.
# MAGIC - `cannot inspect` usually means your Databricks identity has no Postgres
# MAGIC   role on that instance. Create one from the instance's **Permissions** tab.
