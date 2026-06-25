

## 목차

1. [개요](#개요)
2. [Namespace](#1-namespace)
3. [Event](#2-event)
4. [Room](#3-room)
5. [실전 예시](#4-실전-예시)
6. [정리](#5-정리)

---

## 개요

Socket.IO로 실시간 기능을 설계하다 보면 Namespace, Event, Room이라는 세 가지 개념을 마주하게 된다. 이름만 봐서는 역할 구분이 헷갈릴 수 있는데, 각각의 목적이 명확히 다르다.

**Namespace**는 연결 공간을 분리하는 단위이고, **Event**는 클라이언트와 서버가 주고받는 메시지의 이름이며, **Room**은 특정 클라이언트들에게만 메시지를 보내기 위한 서버 측 그룹이다.

| 개념      | 의미                   | 예시                                          |
| --------- | ---------------------- | --------------------------------------------- |
| Namespace | 연결 영역 분리         | `/chat`, `/notification`, `/dashboard`        |
| Event     | 메시지 종류 구분       | `subscribe`, `unsubscribe`, `realtime_data`   |
| Room      | 메시지를 받을 클라이언트 그룹 | `project:123`, `user:45`, `dashboard:sales` |

한 줄로 요약하면 이렇다.

> **Namespace로 기능 영역을 나누고, Room으로 수신 대상을 나누고, Event로 메시지 종류를 구분한다.**

```text
Namespace: /dashboard
Room:      dashboard:sales
Event:     realtime_data
```

위 구조는 `/dashboard` 연결 공간 안에서, `dashboard:sales` 대시보드를 보고 있는 사용자들에게, `realtime_data` 이벤트로 실시간 데이터를 전송하는 방식이다.

---

## 1. Namespace

**Namespace는 Socket.IO 연결 공간을 기능 단위로 분리하는 개념이다.**

기본 Namespace는 `/`다.

```javascript
const socket = io("/");
```

서비스 규모가 커지면 하나의 연결 공간에서 모든 이벤트를 처리하기보다, 기능별로 연결 공간을 나누는 편이 관리하기 훨씬 쉽다.

```javascript
const chatSocket         = io("/chat");
const notificationSocket = io("/notification");
const dashboardSocket    = io("/dashboard");
```

서버에서도 각각의 Namespace를 따로 처리할 수 있다.

```javascript
const chat = io.of("/chat");
chat.on("connection", (socket) => {
  console.log("chat namespace connected");
});

const notification = io.of("/notification");
notification.on("connection", (socket) => {
  console.log("notification namespace connected");
});

const dashboard = io.of("/dashboard");
dashboard.on("connection", (socket) => {
  console.log("dashboard namespace connected");
});
```

Namespace는 다음처럼 **큰 기능 영역을 분리할 때** 사용하는 것이 적합하다.

```text
/chat           채팅 기능
/notification   알림 기능
/dashboard      실시간 대시보드 기능
/admin          관리자 기능
```

> **⚠️ Namespace를 너무 잘게 나누는 것은 피해야 한다.**
>
> 예를 들어 이벤트 하나마다 Namespace를 만들면 연결 관리가 불필요하게 복잡해진다.
>
> ```text
> # 지양
> /message / /read / /typing / /online
>
> # 권장
> /chat  →  내부에서 Event와 Room으로 세부 기능 구분
> ```

---

## 2. Event

**Event는 클라이언트와 서버가 주고받는 메시지의 이름이다.**

클라이언트가 서버로 메시지를 보낼 때도, 서버가 클라이언트에게 메시지를 보낼 때도 모두 Event를 사용한다.

예를 들어 클라이언트가 특정 대시보드 데이터를 구독하고 싶다면 `subscribe` 이벤트를 보낸다.

```javascript
// 클라이언트 → 서버
socket.emit("subscribe", {
  roomId: "dashboard:sales"
});
```

서버는 `subscribe` 이벤트를 받아 처리한다.

```javascript
// 서버에서 수신
socket.on("subscribe", (payload) => {
  console.log(payload.roomId); // "dashboard:sales"
});
```

반대로 서버가 특정 클라이언트에게 실시간 데이터를 전달할 때는 `realtime_data` 같은 이벤트를 사용한다.

```javascript
// 서버 → 특정 Room의 클라이언트 전체에 broadcast
io.of("/dashboard").to("dashboard:sales").emit("realtime_data", {
  type: "sales_count",
  value: 1520,
  timestamp: "2026-06-25T16:00:00+09:00"
});
```

Event는 **"이 메시지가 어떤 목적인가?"** 를 구분하는 이름이다.

```text
subscribe       구독 요청
unsubscribe     구독 해제 요청
realtime_data   실시간 데이터 전달
message         채팅 메시지 전달
typing          입력 중 상태 전달
error           에러 응답
```

---

## 3. Room

**Room은 같은 Namespace 안에서 특정 클라이언트들을 묶는 서버 측 그룹이다.**

예를 들어 여러 사용자가 같은 프로젝트 페이지를 보고 있다면, 서버는 해당 사용자들의 socket을 `project:123` Room에 묶을 수 있다.

```javascript
socket.join("project:123");
```

이후 서버는 `project:123` Room에 속한 클라이언트들에게만 메시지를 보낼 수 있다.

```javascript
io.of("/dashboard")
  .to("project:123")
  .emit("realtime_data", {
    projectId: 123,
    activeUsers: 8,
    updatedTaskCount: 24
  });
```

Room은 다음과 같은 상황에서 유용하다.

```text
project:123         특정 프로젝트 화면을 보고 있는 사용자 그룹
chat:room:10        특정 채팅방에 들어와 있는 사용자 그룹
user:45             특정 사용자에게만 보내는 개인 알림 그룹
dashboard:sales     매출 대시보드를 보고 있는 사용자 그룹
document:777        같은 문서를 공동 편집 중인 사용자 그룹
```

중요한 점이 두 가지 있다.

**첫째, Room은 서버가 관리하는 그룹이다.**

클라이언트는 "이 데이터를 구독하고 싶다"고 요청만 하고, 실제로 어떤 Room에 넣을지는 서버가 결정한다.

**둘째, 클라이언트가 연결을 끊으면 모든 Room에서 자동으로 제거된다.**

따라서 `disconnect` 이벤트에서 직접 `socket.leave()`를 호출하지 않아도 된다. 다만 사용자가 특정 화면을 벗어나는 것처럼 연결은 유지되면서 구독만 해제하는 경우에는 명시적으로 `socket.leave()`를 호출해야 한다.

---

## 4. 실전 예시

실시간 대시보드 기능을 설계하는 상황을 생각해보자.

사용자가 매출 대시보드 화면에 진입하면 해당 데이터를 구독하고, 화면을 벗어나면 구독을 해제하는 흐름이다.

```text
Namespace: /dashboard
Room:      dashboard:sales
Events:    subscribe / unsubscribe / realtime_data
```

### 클라이언트

```javascript
const socket = io("/dashboard");

// 매출 대시보드 화면 진입 시 구독 요청
socket.emit("subscribe", {
  roomId: "dashboard:sales"
});

// 구독 완료 확인 수신
socket.on("subscribe_success", (data) => {
  console.log(data.message); // "구독이 완료되었습니다."
});

// 실시간 데이터 수신 → 화면 갱신
socket.on("realtime_data", (data) => {
  console.log("실시간 데이터 수신:", data);
  // 1. 매출 금액 업데이트
  // 2. 주문 수 업데이트
  // 3. 실시간 차트 갱신
});

// 매출 대시보드 화면 이탈 시 구독 해제 요청
socket.emit("unsubscribe", {
  roomId: "dashboard:sales"
});

// 구독 해제 완료 확인 수신
socket.on("unsubscribe_success", (data) => {
  console.log(data.message); // "구독이 해제되었습니다."
});
```

### 서버

```javascript
const dashboard = io.of("/dashboard");

dashboard.on("connection", (socket) => {

  // 구독 요청 처리
  socket.on("subscribe", (payload) => {
    const { roomId } = payload;

    socket.join(roomId); // Room에 추가

    socket.emit("subscribe_success", {
      roomId,
      message: "구독이 완료되었습니다."
    });
  });

  // 구독 해제 요청 처리
  socket.on("unsubscribe", (payload) => {
    const { roomId } = payload;

    socket.leave(roomId); // Room에서 제거

    socket.emit("unsubscribe_success", {
      roomId,
      message: "구독이 해제되었습니다."
    });
  });

});

// 매출 데이터 변경 시 해당 Room 전체에 broadcast
io.of("/dashboard")
  .to("dashboard:sales")
  .emit("realtime_data", {
    type: "sales_summary",
    todaySales: 1250000,
    orderCount: 320,
    timestamp: "2026-06-25T16:00:00+09:00"
  });
```

### 전체 흐름 요약

```text
1. 클라이언트가 /dashboard Namespace에 연결한다.
2. 사용자가 매출 대시보드 화면에 진입한다.
3. 클라이언트가 subscribe 이벤트로 dashboard:sales 구독을 요청한다.
4. 서버가 해당 socket을 dashboard:sales Room에 추가하고 subscribe_success를 응답한다.
5. 서버는 매출 데이터가 변경될 때마다 dashboard:sales Room 전체에 realtime_data를 broadcast한다.
6. 클라이언트는 realtime_data 이벤트를 받아 화면을 갱신한다.
7. 사용자가 화면을 이탈하면 unsubscribe 이벤트를 보낸다.
8. 서버가 해당 socket을 dashboard:sales Room에서 제거하고 unsubscribe_success를 응답한다.
```

각 개념의 역할을 정리하면 다음과 같다.

| 개념                     | 실제 역할                              |
| ------------------------ | -------------------------------------- |
| `/dashboard` Namespace   | 대시보드 기능 전용 연결 공간           |
| `dashboard:sales` Room   | 매출 대시보드를 보고 있는 사용자 그룹  |
| `subscribe` Event        | 특정 대시보드 데이터 구독 요청         |
| `unsubscribe` Event      | 특정 대시보드 데이터 구독 해제 요청    |
| `realtime_data` Event    | 서버가 Room 전체에 전송하는 실시간 데이터 |

---

## 5. 정리

Socket.IO에서 Namespace, Event, Room은 각각 역할이 다르다.

- **Namespace** — 연결 영역을 기능 단위로 분리한다. `/chat`, `/notification`, `/dashboard`처럼 큰 기능 경계에서만 나눈다.
- **Event** — 메시지의 목적을 구분하는 이름이다. `subscribe`, `unsubscribe`, `realtime_data`처럼 "이 메시지가 무엇을 하는가"를 표현한다.
- **Room** — 서버가 관리하는 클라이언트 그룹이다. `dashboard:sales`, `project:123`처럼 메시지를 받을 대상을 묶는다.

설계 기준을 한 줄로 정리하면 이렇다.

```text
기능 단위   →  Namespace
수신 대상   →  Room
메시지 목적 →  Event
```

이 세 가지 개념을 역할에 맞게 사용하면, 복잡한 실시간 기능도 명확하게 구조화할 수 있다.
