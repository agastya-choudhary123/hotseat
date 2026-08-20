# A worker in a container.
#
# Two of these on one Docker network are the cross-container migration demo.
# The base is the same rust:1-bookworm the mincontainer dev image uses, so the
# build environment matches the one that project already runs Linux work in.
FROM rust:1-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release --bin hs-worker --bin hotseat \
 && cargo build --release --example faultbench -p hs-track \
 && cargo test  --release --no-run -p hs-track

# Keep the test binary so the container can prove its own tracker works.
RUN mkdir -p /out && cp target/release/hs-worker target/release/hotseat /out/ \
 && cp target/release/examples/faultbench /out/ \
 && cp "$(ls -t target/release/deps/tracker-* | grep -v '\.d$' | head -1)" /out/tracker-tests

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends procps \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /usr/local/bin/
ENV HOTSEAT_MODEL=/models/model.gguf
EXPOSE 7400
ENTRYPOINT ["/usr/local/bin/hs-worker"]
