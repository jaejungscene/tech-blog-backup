## 목차
- [1. Kafka Partition이 필요한 이유](#1-kafka-partition이-필요한-이유)
- [2. RabbitMQ에서 Partition 기능을 대체할 두 가지 대안](#2-rabbitmq에서-partition-기능을-대체할-두-가지-대안)
- [3. x-modulus-hash 기반 전체 구조](#3-x-modulus-hash-기반-전체-구조)
- [4. 기존 구조를 partition 구조로 바꾸기](#4-기존-구조를-partition-구조로-바꾸기)
- [5. Python 구현 예시](#5-python-구현-예시)
    - [topology 생성](#topology-생성)
    - [Producer](#producer)
    - [Consumer](#consumer)
    - [실행](#실행)
- [6. 설계 기준: routing key, Queue 수, consumer 수](#6-설계-기준-routing-key-queue-수-consumer-수)
    - [routing key를 무엇으로 쓸 것인가](#routing-key를-무엇으로-쓸-것인가)
    - [Queue 개수](#queue-개수)
    - [Queue당 consumer 수](#queue당-consumer-수)
- [7. 운영 환경에서 주의할 점](#7-운영-환경에서-주의할-점)
    - [Queue 개수를 자주 바꾸지 않는다](#queue-개수를-자주-바꾸지-않는다)
    - [Hot key 문제](#hot-key-문제)
    - [실패 처리](#실패-처리)
    - [메시지 자체에 순서 검증 정보를 넣는다](#메시지-자체에-순서-검증-정보를-넣는다)
- [8. 마무리](#8-마무리)


## 1. Kafka Partition이 필요한 이유

Kafka의 Partition은 하나의 Topic 데이터를 여러 갈래로 나누어 병렬 처리하면서도, 특정 기준의 순서를 보장하고 싶을 때 사용한다.

`order_events`라는 Topic에 아래 이벤트들이 들어온다고 하자.

```text
주문 생성 → 결제 완료 → 배송 요청
```

`ORD-1001`이라는 주문이 있다면 이 세 이벤트는 반드시 이 순서로 처리되어야 한다. 하지만 여러 consumer가 `order_id` 기준 없이 이벤트를 나누어 처리하면, 같은 주문의 이벤트가 서로 다른 consumer에서 처리되면서 순서가 깨질 수 있다.

```mermaid
sequenceDiagram
    autonumber
    participant T as order_events Topic
    participant P0 as partition-0
    participant P1 as partition-1
    participant C0 as consumer-0
    participant C1 as consumer-1

    Note over T,C1: order_id 기준 없이 분산되는 경우

    T->>P1: ORD-1001 주문 생성
    T->>P0: ORD-1001 결제 완료
    T->>P1: ORD-1001 배송 요청

    P1->>C1: ORD-1001 주문 생성
    P0->>C0: ORD-1001 결제 완료
    P1->>C1: ORD-1001 배송 요청

    Note over C0,C1: 같은 order_id 이벤트가 서로 다른 consumer에서 처리됨
```

물론 Redis 분산 락이나 DB 상태 체크, 재시도 큐를 조합해 애플리케이션 레벨에서 순서를 맞출 수도 있다. 하지만 그렇게 하면 메시징 시스템이 해결해줘야 할 문제를 애플리케이션 코드가 떠안게 되고, 장애 원인 추적도 어려워진다.

Kafka는 이 문제를 Partition Key로 해결한다. Partition Key를 `order_id`로 지정하면, 같은 `order_id`를 가진 메시지는 항상 같은 partition으로 들어가고 partition 내부의 순서는 보장된다.

```mermaid
sequenceDiagram
    autonumber
    participant T as order_events Topic
    participant P0 as partition-0
    participant P1 as partition-1
    participant C0 as consumer-0
    participant C1 as consumer-1

    Note over T,C1: key = order_id 로 분산하는 경우

    T->>P0: key=ORD-1001, 주문 생성
    T->>P0: key=ORD-1001, 결제 완료
    T->>P0: key=ORD-1001, 배송 요청

    T->>P1: key=ORD-1002, 주문 생성
    T->>P1: key=ORD-1002, 결제 완료
    T->>P1: key=ORD-1002, 배송 요청

    P0->>C0: ORD-1001 이벤트 순차 처리
    P1->>C1: ORD-1002 이벤트 순차 처리

    Note over C0,C1: 전체 처리는 병렬화하고, order_id별 순서는 유지
```

이 구조가 유용한 곳은 주문 도메인만이 아니다.

```text
- order_id별 주문 상태 변경 이벤트
- user_id별 사용자 활동 이벤트
- account_id별 잔액 변경 이벤트
- device_id / equipment_id별 장비·설비 상태 이벤트
- room_id별 채팅 메시지 이벤트
```

공통점은 하나다. **전체 이벤트는 병렬로 빠르게 처리하되, 같은 business key의 이벤트는 순서대로 처리하고 싶다**는 것. Kafka Partition의 핵심은 단순히 데이터를 나누는 게 아니라, key 기준으로 데이터를 나눠서 병렬 처리와 순서 보장을 동시에 가져가는 데 있다.

물론 RabbitMQ는 Kafka처럼 "스트림 처리 플랫폼"으로서의 목적을 두고 탄생한 것이 아니라, 애플리케이션 간 메시지를 안정적으로 라우팅하고 전달하기 위한 메시지 브로커로서의 목적을 가지고 탄생했기에 Partition이라는 개념이 없다. 하지만 위와 같은 요구사항을 충족시킬 수 있는 기능적 대안은 존재한다 (다만 완전히 Partition이랑 동일하지는 않다). 따라서 위와 같은 요구사항이 있을 때, 이미 Kafka 혹은 NATS Jetstream과 같은 스트림 처리 플랫폼을 사용하고 있다면, 그냥 Partition을 사용하여 구현하면 된다. 하지만 나와 같이 이미 회사의 소프트웨어가 RabbitMQ 플랫폼으로 구현되어 있고, 여러 비용(전환 비용·시간·유지보수 비용 등)을 고려했을 때, 스트림 처리 플랫폼으로 마이그레이션 할 수 없는 상황에 놓인 분들이라면 이 글을 참고하면 좋을 것 같다.

## 2. RabbitMQ에서 Partition 기능을 대체할 두 가지 대안

RabbitMQ에서 비슷한 구조를 만들려면 두 exchange type 중 하나를 선택해야 한다.

```text
1. x-consistent-hash
2. x-modulus-hash
```

두 방식 모두 메시지의 key를 hash해서 여러 Queue 중 하나로 라우팅한다는 점은 같다. 하지만 **고정된 partition 구조**를 만드는 게 목적이라면 둘은 꽤 다르게 동작한다.

**x-consistent-hash**는 consistent hashing 기반으로 Queue 사이에 메시지를 분산하고, Queue별로 weight를 줄 수 있다. Queue 추가/삭제 시 key 이동을 최소화하는 것이 consistent hashing 본연의 장점이다. 다만 RabbitMQ 팀이 직접 밝힌 바에 따르면, hash ring은 메모리(ETS)에 저장되고 binding 정보로부터 재구성되는데, 바인딩이 동일하게 복원되더라도 ring 위에서 각 Queue가 차지하는 위치가 이전과 달라질 수 있어 routing key → Queue 매핑이 재시작 전후로 항상 동일하다고 보장하지 않는다. 즉 "같은 key는 항상 같은 partition Queue로 간다"는 Kafka식 사고와는 다소 거리가 있다.

**x-modulus-hash**는 Producer가 publish할 때 사용한 routing key를 hash한 뒤 `Hash mod N`으로 Queue를 선택한다. 여기서 `N`은 exchange에 bind된 Queue 개수다.

```text
Kafka
hash(order_id) % partition_count → 특정 partition

RabbitMQ x-modulus-hash
hash(order_id) % bound_queue_count → 특정 Queue
```

바인딩 구성이 바뀌지 않는 한 같은 key는 항상 같은 Queue로 라우팅되므로, "같은 key를 같은 처리 단위로 보내고 그 단위별로 consumer를 붙인다"는 목적에 더 잘 맞는다. 이 글에서는 `x-modulus-hash`를 기준으로 구조를 정리한다.

> **버전 참고**
> `x-modulus-hash`는 원래 `rabbitmq_sharding` plugin이 제공하던 exchange type이다. RabbitMQ 4.3.0(2026년 4월 출시)부터는 이 exchange type이 core로 흡수되었고, 안정적인 routing(바인딩 구성이 그대로라면 노드 재시작 후에도 동일하게 routing되는 것)을 제공하도록 재구현되었다. 따라서 RabbitMQ 4.3 이상을 쓰고 있다면 별도 plugin 설치/활성화 없이 바로 `exchange_type="x-modulus-hash"`로 선언해 쓸 수 있다.
>
> RabbitMQ 4.3 미만 버전이라면 여전히 plugin을 활성화해야 한다. 이 plugin은 3.6.0 이후 공식 distribution에 포함되어 있다.
> ```bash
> rabbitmq-plugins enable rabbitmq_sharding   # RabbitMQ 4.3 미만에서만 필요
> ```
> 운영 환경의 RabbitMQ 버전을 먼저 확인하고 이 단계를 넣을지 말지 정하는 것이 좋다.

## 3. x-modulus-hash 기반 전체 구조

```mermaid
flowchart LR
    P[Producer] -->|routing_key = order_id| X[x-modulus-hash exchange]

    X -->|hash mod N| Q0[order_events.partition.0]
    X -->|hash mod N| Q1[order_events.partition.1]
    X -->|hash mod N| Q2[order_events.partition.2]
    X -->|hash mod N| Q3[order_events.partition.3]
    X -->|hash mod N| Q4[order_events.partition.4]

    Q0 --> C0[consumer-0]
    Q1 --> C1[consumer-1]
    Q2 --> C2[consumer-2]
    Q3 --> C3[consumer-3]
    Q4 --> C4[consumer-4]
```

핵심은 단순하다.

```text
1. x-modulus-hash exchange를 만든다.
2. partition 역할을 할 Queue 여러 개를 exchange에 bind한다.
3. Producer는 routing_key에 order_id를 넣어 publish한다.
4. 각 Queue는 하나의 consumer가 순차 처리한다.
```

`ORD-1001`의 모든 이벤트는 항상 같은 Queue로 들어가지만, 다른 `order_id`는 hash 결과에 따라 다른 Queue로 분산된다.

```text
ORD-1001 → order_events.partition.2 (항상 동일)
ORD-1002 → order_events.partition.0
ORD-1003 → order_events.partition.4
```

이렇게 하면 전체 처리량은 Queue 개수만큼 병렬화하면서도, 같은 `order_id`의 이벤트는 같은 Queue에서 순서대로 처리할 수 있다.

## 4. 기존 구조를 partition 구조로 바꾸기

기존 시스템이 아래처럼 Queue 하나에 여러 consumer가 붙어 있는 구조였다고 하자.

```mermaid
flowchart LR
    P[Producer] --> X[order_events exchange]
    X --> Q[order_events queue]
    Q --> C1[consumer-1]
    Q --> C2[consumer-2]
    Q --> C3[consumer-3]
```

처리량은 높일 수 있지만, 같은 `order_id`의 이벤트가 서로 다른 consumer에게 나뉘어 처리되면서 순서가 깨질 수 있다.

```mermaid
sequenceDiagram
    autonumber
    participant Q as order_events queue
    participant C1 as consumer-1
    participant C2 as consumer-2
    participant C3 as consumer-3

    Q->>C1: ORD-1001 주문 생성
    Q->>C2: ORD-1001 결제 완료
    Q->>C3: ORD-1001 배송 요청

    Note over C1,C3: 같은 order_id 이벤트가 서로 다른 consumer에서 처리됨
```

이 구조를 `x-modulus-hash` 기반으로 바꾸면, 기존 `order_events queue` 하나를 여러 partition Queue로 나누고 각 Queue에 consumer를 하나씩 붙인다.

```mermaid
flowchart LR
    P[Producer] -->|routing_key = order_id| X["x-modulus-hash<br/>order_events.partitioned"]

    X --> Q0[order_events.partition.0]
    X --> Q1[order_events.partition.1]
    X --> Q2[order_events.partition.2]
    X --> Q3[order_events.partition.3]
    X --> Q4[order_events.partition.4]

    Q0 --> C0[consumer-0]
    Q1 --> C1[consumer-1]
    Q2 --> C2[consumer-2]
    Q3 --> C3[consumer-3]
    Q4 --> C4[consumer-4]
```

```text
Before
- 하나의 order_events queue에 여러 consumer가 붙음
- 같은 order_id 이벤트가 서로 다른 consumer에서 처리될 수 있음

After
- x-modulus-hash exchange 사용
- order_events.partition.N Queue 여러 개 생성
- routing_key = order_id 로 publish
- 같은 order_id는 같은 Queue로 라우팅
- Queue별 consumer가 순차 처리
```

## 5. Python 구현 예시

`pika`를 사용한다.

```bash
pip install pika
```

#### topology 생성

`x-modulus-hash` exchange와 partition Queue들을 만든다. 처리에 실패한 메시지를 담을 DLX(fanout exchange)와 DLQ도 함께 선언해서, partition Queue에 `x-dead-letter-exchange`를 지정해둔다.

```python
# setup_topology.py

import pika

RABBITMQ_URL = "amqp://guest:guest@localhost:5672/%2F"

EXCHANGE_NAME = "order_events.partitioned"
QUEUE_PREFIX = "order_events.partition"
PARTITION_COUNT = 5

DLX_NAME = "order_events.dlx"
DLQ_NAME = "order_events.dlq"


def main():
    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    # 1. x-modulus-hash exchange 선언
    channel.exchange_declare(
        exchange=EXCHANGE_NAME,
        exchange_type="x-modulus-hash",
        durable=True,
    )

    # 1-1. 실패 메시지를 받을 DLX(fanout)와 DLQ 선언
    # partition Queue에서 nack(requeue=False)한 메시지는 이 DLX를 거쳐 DLQ에 쌓인다.
    channel.exchange_declare(
        exchange=DLX_NAME,
        exchange_type="fanout",
        durable=True,
    )
    channel.queue_declare(queue=DLQ_NAME, durable=True)
    channel.queue_bind(exchange=DLX_NAME, queue=DLQ_NAME, routing_key="")

    # 2. partition 역할을 할 Queue 여러 개 선언
    for i in range(PARTITION_COUNT):
        queue_name = f"{QUEUE_PREFIX}.{i}"

        channel.queue_declare(
            queue=queue_name,
            durable=True,
            arguments={
                # Queue에 백업 consumer를 여러 개 붙여두더라도
                # 실제 active consumer는 하나만 동작하도록 보장한다.
                # consumer를 정말 하나씩만 띄울 거라면 필수는 아니지만,
                # failover를 고려해 기본적으로 켜두는 편이 안전하다.
                "x-single-active-consumer": True,
                # 처리 실패한 메시지(nack, requeue=False)를 DLX로 보낸다.
                "x-dead-letter-exchange": DLX_NAME,
            },
        )

        # 3. exchange에 Queue binding
        # x-modulus-hash는 binding key를 라우팅 기준으로 쓰지 않는다.
        # 메시지 publish 시 전달되는 routing_key만 hash 기준으로 사용한다.
        channel.queue_bind(
            exchange=EXCHANGE_NAME,
            queue=queue_name,
            routing_key="",
        )

    connection.close()


if __name__ == "__main__":
    main()
```

`x-modulus-hash`는 Queue를 bind할 때 사용한 binding key를 완전히 무시하고, Producer가 publish 시 전달한 `routing_key`만 hash한다. 따라서 실제 partition key는 Producer 코드에서 결정된다.

#### Producer

```python
# producer.py

import json
import uuid
from datetime import datetime, timezone

import pika

RABBITMQ_URL = "amqp://guest:guest@localhost:5672/%2F"
EXCHANGE_NAME = "order_events.partitioned"


def publish_order_event(order_id: str, event_type: str, payload: dict, order_version: int = 1):
    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    message = {
        "event_id": str(uuid.uuid4()),
        "order_id": order_id,
        "event_type": event_type,
        # 7장에서 설명하는 순서 검증용 필드. 실무에서는 주문 상태를 조회해
        # 다음 order_version 값을 채워야 한다. 여기서는 단순화를 위해
        # 호출하는 쪽에서 직접 지정한다.
        "order_version": order_version,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }

    body = json.dumps(message, ensure_ascii=False).encode("utf-8")

    channel.basic_publish(
        exchange=EXCHANGE_NAME,
        # Kafka partition key처럼 사용할 값을 routing_key에 넣는다.
        # 같은 order_id는 같은 Queue로 라우팅된다.
        routing_key=order_id,
        body=body,
        properties=pika.BasicProperties(
            content_type="application/json",
            delivery_mode=2,  # persistent message
            message_id=message["event_id"],
            headers={"event_type": event_type},
        ),
    )

    connection.close()


if __name__ == "__main__":
    publish_order_event("ORD-1001", "ORDER_CREATED", {"amount": 30000}, order_version=1)
    publish_order_event("ORD-1001", "PAYMENT_COMPLETED", {"payment_id": "PAY-777"}, order_version=2)
    publish_order_event("ORD-1001", "SHIPPING_REQUESTED", {"address": "Seoul"}, order_version=3)
    publish_order_event("ORD-1002", "ORDER_CREATED", {"amount": 50000}, order_version=1)
```

#### Consumer

partition Queue 하나를 담당하도록 작성한다.

```python
# consumer.py

import json
import sys
import time

import pika

RABBITMQ_URL = "amqp://guest:guest@localhost:5672/%2F"
QUEUE_PREFIX = "order_events.partition"


def process_message(message: dict):
    print(f"[process] order_id={message['order_id']}, event_type={message['event_type']}")
    # 실제 비즈니스 로직: 주문 상태 변경, 결제 상태 반영, 배송 요청 생성 등
    time.sleep(0.2)


def main(partition_index: int):
    queue_name = f"{QUEUE_PREFIX}.{partition_index}"

    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    # 순서 보장을 단순하게 가져가려면 prefetch_count=1이 가장 안전하다.
    # 메시지 하나를 처리하고 ack한 뒤에야 다음 메시지를 받는다.
    channel.basic_qos(prefetch_count=1)

    def callback(ch, method, properties, body):
        try:
            message = json.loads(body.decode("utf-8"))
            process_message(message)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as exc:
            print(f"[error] failed to process message: {exc}")
            # 운영 환경에서는 무한 requeue보다 DLQ를 권장한다.
            # 이 Queue는 setup_topology.py에서 x-dead-letter-exchange가
            # 설정되어 있으므로, requeue=False로 nack하면
            # order_events.dlq로 전달된다.
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=False)

    print(f"Consuming from {queue_name}")
    channel.start_consuming()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python consumer.py <partition_index>")
        sys.exit(1)

    main(partition_index=int(sys.argv[1]))
```

#### 실행

```bash
python setup_topology.py

python consumer.py 0
python consumer.py 1
python consumer.py 2
python consumer.py 3
python consumer.py 4
```

## 6. 설계 기준: routing key, Queue 수, consumer 수

`x-modulus-hash` 자체는 옵션이 많지 않다. 핵심 동작은 `hash(publish routing_key) % bound_queue_count` 하나뿐이고, 실무에서 신경 써야 할 것은 아래 세 가지다.

#### routing key를 무엇으로 쓸 것인가

같은 값끼리 순서를 지켜야 하는 값을 routing key로 쓰면 된다. 주문 도메인이면 `order_id`, 사용자별 순서가 중요하면 `user_id`, 설비별 순서가 중요하면 `equipment_id`를 쓰는 식이다.

```python
channel.basic_publish(exchange="order_events.partitioned", routing_key=order_id, body=body)
channel.basic_publish(exchange="user_events.partitioned", routing_key=user_id, body=body)
channel.basic_publish(exchange="equipment_events.partitioned", routing_key=equipment_id, body=body)
```

#### Queue 개수

Kafka의 partition count와 비슷하게 생각하면 된다. Queue가 5개면 `hash(routing_key) % 5`로 분산된다.

```text
Queue 수가 너무 적으면 → 특정 Queue가 병목이 될 수 있다.
Queue 수가 너무 많으면 → consumer 운영, 모니터링, 장애 대응이 복잡해진다.
```

당장 consumer가 5개라도, 나중에 늘릴 가능성이 있다면 처음부터 여유 있게 Queue 수를 잡아둘 수 있다. 다만 partition Queue가 N개라면 모든 메시지를 소비하기 위해 최소 N개의 consumer가 붙어 있어야 한다는 점은 그대로다. 일부 Queue에 consumer가 붙어 있지 않으면 그 Queue에 쌓인 메시지는 처리되지 않는다.

#### Queue당 consumer 수

같은 `order_id`가 같은 Queue로 들어가더라도, 그 Queue에 여러 consumer가 동시에 붙으면 처리 순서가 깨질 수 있다. 따라서 아래 둘 중 하나를 선택해야 한다.

```text
방법 1. Queue 하나당 consumer 하나만 실행한다.
방법 2. x-single-active-consumer 옵션을 사용한다.
```

`x-single-active-consumer`를 켜두면 Queue에 여러 consumer가 붙어 있어도 한 번에 하나만 active하게 메시지를 소비하고, active consumer에 문제가 생기면 다른 consumer가 이어받는다. consumer 쪽에서는 `prefetch_count=1`을 쓰는 것이 순서 보장을 설명하고 구현하기에 가장 단순하다. 처리량이 더 필요하면 `prefetch_count`를 늘릴 수 있지만, 그 경우 consumer 내부에서 메시지를 병렬로 처리하지 않도록 주의해야 한다. 순서대로 들어온 메시지를 애플리케이션 내부에서 병렬 처리해버리면 순서가 다시 뒤집힐 수 있다.

## 7. 운영 환경에서 주의할 점

#### Queue 개수를 자주 바꾸지 않는다

`x-modulus-hash`는 `Hash mod N` 구조이고, `N`은 bind된 Queue 개수다. Queue 개수가 바뀌면 계산식 자체가 바뀌어 많은 key의 매핑이 한꺼번에 바뀐다.

```text
hash(ORD-1001) % 5 = 2   → order_events.partition.2
hash(ORD-1001) % 6 = 4   → order_events.partition.4 (Queue를 6개로 늘린 경우)
```

Kafka에서 partition 수를 함부로 바꾸지 않는 것과 같은 이유다. partition Queue 수는 처음에 신중히 정하고 운영 중에는 가급적 바꾸지 않는 것이 좋다.

#### Hot key 문제

`order_id`가 충분히 다양하면 메시지는 여러 Queue에 비교적 고르게 분산되지만, 특정 key에 메시지가 몰리면 그 Queue 하나가 병목이 된다. Kafka의 hot partition 문제와 동일하다. order_id별 strict ordering이 꼭 필요하다면 이 병목은 감수해야 하고, 순서 보장이 덜 중요하다면 key를 더 세분화하거나 병렬 처리 가능한 단계와 순차 처리 단계를 분리하는 식으로 풀어야 한다.

#### 실패 처리

순서가 중요한 Queue에서 실패 메시지를 무한 requeue하면 뒤따르는 메시지 처리가 막히거나 전체 partition이 멈출 수 있다.

```python
# 피해야 할 패턴
ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
```

대신 아래 중 하나를 선택하는 것이 안전하다.

```text
1. 일정 횟수 재시도 후 DLQ로 이동
2. 실패한 order_id를 별도 보정 프로세스로 넘김
3. consumer를 멈추고 운영자가 원인 확인 후 재개
```

5장 예시 코드에서는 partition Queue에 `x-dead-letter-exchange`를 지정해두었기 때문에, `nack(requeue=False)`만으로 실패 메시지가 자동으로 `order_events.dlq`에 쌓인다. 다만 예시 코드는 재시도 횟수를 세지 않고 첫 실패에 바로 DLQ로 보내는 단순한 형태이므로, 일시적 오류(네트워크 지연 등)까지 걸러내려면 재시도 횟수 카운트를 추가하는 것이 좋다.

#### 메시지 자체에 순서 검증 정보를 넣는다

메시징 시스템이 같은 key를 같은 Queue로 보내주더라도, Producer가 잘못된 순서로 메시지를 발행하면 RabbitMQ가 그 순서를 자동으로 고쳐주지는 않는다. 중요한 도메인이라면 메시지에 version이나 sequence를 넣어두는 것이 좋다. 5장 producer.py 예시에도 이 `order_version` 필드를 포함해두었다.

```json
{
  "event_id": "4d85f4c1-53c2-4c3f-9de7-6f5d74c91c71",
  "order_id": "ORD-1001",
  "event_type": "PAYMENT_COMPLETED",
  "order_version": 2,
  "occurred_at": "2026-06-30T12:00:00Z",
  "payload": { "payment_id": "PAY-777" }
}
```

Consumer는 `order_version`을 보고 이미 처리한 이벤트인지, 이전 version을 건너뛰고 들어온 이벤트인지, 현재 주문 상태에서 처리 가능한 이벤트인지를 검증할 수 있다. 메시징 레벨에서는 "같은 key를 같은 Queue로 보내는 것"까지만 담당하고, 도메인 정합성은 애플리케이션에서 한 번 더 검증하는 것이 안전하다.

## 8. 마무리

RabbitMQ에서 Kafka Partition과 비슷한 구조가 필요하다면 `x-modulus-hash`를 중심으로 설계하는 것이 가장 직관적이다.

```text
Producer → x-modulus-hash exchange → partition 역할을 하는 여러 Queue → Queue별 single active consumer
```

`routing_key`에 `order_id`를 넣으면 같은 주문 이벤트는 같은 Queue로 들어가 순차 처리된다. 운영에서는 아래 원칙을 지키는 것이 중요하다.

```text
1. routing_key는 순서를 보장하고 싶은 business key로 정한다.
2. partition Queue 개수는 처음에 신중하게 정하고 자주 바꾸지 않는다.
3. Queue 하나는 한 번에 하나의 active consumer만 처리하게 한다.
4. 실패 메시지는 무한 requeue하지 말고 DLQ나 보정 흐름으로 분리한다.
5. 중요한 도메인 이벤트에는 version/sequence를 넣어 애플리케이션에서도 검증한다.
6. RabbitMQ 4.3 이상이면 plugin 없이 바로 쓸 수 있다는 점을 확인하고 시작한다.
```
