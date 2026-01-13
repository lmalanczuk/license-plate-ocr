from redis import Redis
from rq import Queue
from rq.worker import SimpleWorker

redis_conn = Redis(host="localhost", port=6379)
queue = Queue("plates", connection=redis_conn)

if __name__ == "__main__":
    worker = SimpleWorker([queue], connection=redis_conn)
    worker.work()
