from dask_gateway import Gateway

def setup_dask_distributed(flag_shut=False,
                           adapt_min=15, adapt_max=40):
    gateway = Gateway()

    # close existing clusters
    open_clusters = gateway.list_clusters()
    print(list(open_clusters))

    if flag_shut: 
        if len(open_clusters)>0:
            for c in open_clusters:
                cluster = gateway.connect(c.name)
                cluster.shutdown()  


    options = gateway.cluster_options()

    #options.worker_memory = 16 # 24 works fine for long term mean, but seasonal needs a bit more
    # options.worker_cores = 8

    options.environment = dict(
        DASK_DISTRIBUTED__SCHEDULER__WORKER_SATURATION="1.0"
                                 )

    # Create a cluster with those options
    cluster = gateway.new_cluster(options)
    client = cluster.get_client()
    cluster.adapt(adapt_min, adapt_max)

    return client, gateway