from datetime import datetime, timedelta, timezone, date
from decimal import Decimal
import pytest

from terminalweave.marine.tide import TideSample, TideCurve, UnderKeelRequirement
from terminalweave.marine.berth import Berth, VesselCall, BerthPlanner
from terminalweave.cranes.scheduler import Crane, CraneJob, CraneScheduler
from terminalweave.yard.slotting import YardSlot, YardContainer, YardInventory, ContainerSize
from terminalweave.gate.appointments import GateSlotPolicy, GateBook
from terminalweave.customs.holds import CustomsLedger, HoldRecord
from terminalweave.reefer.monitoring import ReeferPolicy, ReeferMonitor, ReeferReading, AlarmState
from terminalweave.hazmat.compatibility import CompatibilityMatrix, SegregationChecker, HazmatUnit
from terminalweave.hazmat.emergency import EmergencyGuide, EmergencyGuidebook
from terminalweave.hazmat.isolation import (IsolationController, IncidentLevel, ActionState,
    AdjacentContainer, ResponseResource)
from terminalweave.hazmat.placards import DeclaredHazmat, PlacardObservation
from terminalweave.core.errors import StateTransitionError, ValidationError
from terminalweave.equipment.maintenance import EquipmentMeter, MaintenancePlan, MaintenanceManager
from terminalweave.rail.blocks import Wagon, RailContainer, RailBlockPlanner
from terminalweave.billing.demurrage import BusinessCalendar, TariffBand, DemurrageTariff, DemurrageCalculator
from terminalweave.core.money import Money
from terminalweave.transshipment.connections import ServiceCall, Connection, ConnectionGraph

UTC=timezone.utc
D=datetime(2026,1,1,tzinfo=UTC)

def test_tide_interpolation_and_clearance():
    curve=TideCurve([TideSample(D,0.0),TideSample(D+timedelta(hours=2),2.0)])
    req=UnderKeelRequirement(10.0,9.5,1.0)
    assert curve.height_at(D+timedelta(hours=1)) == pytest.approx(1.0)
    assert curve.clearance_m(D+timedelta(hours=1),req) == pytest.approx(1.5)
    assert curve.is_safe(D+timedelta(hours=1),req)

def test_berth_planner_separates_overlapping_calls():
    planner=BerthPlanner(Berth("B1",500,16),safety_margin_m=10)
    a=VesselCall("A",180,30,D,D+timedelta(hours=4),10)
    b=VesselCall("B",180,30,D+timedelta(hours=1),D+timedelta(hours=3),10)
    aa=planner.assign(a); bb=planner.assign(b)
    assert not aa.conflicts(bb)

def test_crane_scheduler_serializes_same_crane():
    s=CraneScheduler([Crane("C1",0,30),Crane("C2",1,25)])
    j1=CraneJob("J1","V1",10,30,D,D+timedelta(hours=4))
    j2=CraneJob("J2","V1",15,20,D,D+timedelta(hours=4))
    a=s.schedule(j1);b=s.schedule(j2)
    assert a.job_id=="J1" and b.job_id=="J2"

def test_yard_requires_support_below():
    inv=YardInventory([YardSlot("A",1,1,1,40000),YardSlot("A",1,1,2,40000)])
    top=YardContainer("TOP",ContainerSize.FT20,10000,"X")
    assert not inv.can_place(top,inv.slots[("A",1,1,2)])

def test_gate_cancel_returns_capacity():
    policy=GateSlotPolicy(30,1,1,1);book=GateBook(policy)
    book.book("A","T1",D)
    assert book.capacity_remaining(D)==0
    book.cancel("A")
    assert book.capacity_remaining(D)==1

def test_customs_hold_blocks_external_movement():
    ledger=CustomsLedger();ledger.place_hold(HoldRecord("H1","CONT","CBP","EXAM",D))
    assert ledger.is_blocked("CONT")
    assert ledger.movement_allowed("CONT","inspection_move")
    assert not ledger.movement_allowed("CONT","gate_out")

def test_reefer_alarm_triggers_after_duration():
    policy=ReeferPolicy(-1,1,timedelta(minutes=10),timedelta(minutes=5),0.2)
    m=ReeferMonitor(policy)
    assert m.ingest(ReeferReading(D,3))==AlarmState.PENDING
    assert m.ingest(ReeferReading(D+timedelta(minutes=11),3))==AlarmState.ACTIVE

def test_hazmat_matrix_checks_declared_pair():
    matrix=CompatibilityMatrix();matrix.set_compatible("3","8",10)
    checker=SegregationChecker(matrix)
    a=HazmatUnit("A","3","Z",0,0);b=HazmatUnit("B","8","Z",11,0)
    assert checker.check_pair(a,b)[0]

def _isolation_controller():
    guides=EmergencyGuidebook([
        EmergencyGuide("3",50,100,False,"foam"),
        EmergencyGuide("4.3",150,300,True,"dry_powder"),
    ])
    matrix=CompatibilityMatrix();matrix.set_compatible("3","8",25)
    return IsolationController(guides,matrix)

def _declared(cid="C1",classes=("3",)):
    return DeclaredHazmat(cid,frozenset(classes),frozenset({"UN1993"}))

def test_isolation_uses_worst_case_when_evidence_conflicts():
    c=_isolation_controller()
    observed=PlacardObservation("C1",frozenset({"4.3"}),frozenset({"UN1993"}))
    inc=c.open_incident("I1","C1",_declared(),observed,[],[],D,"ops")
    plan=inc.plan
    assert plan.conflicting_evidence
    assert plan.hazard_classes==frozenset({"3","4.3"})
    assert plan.isolation_m==150 and plan.evacuation_m==300 and plan.avoid_water

def test_isolation_filters_resources_and_orders_moves():
    c=_isolation_controller()
    resources=[ResponseResource("E-WATER","water"),ResponseResource("E-FOAM","foam"),
               ResponseResource("E-DOWN","foam",available=False)]
    inc=c.open_incident("I1","C1",_declared(classes=("4.3",)),None,[],resources,D,"ops")
    plan=inc.plan
    assert plan.usable_resources==("E-FOAM",)

def test_isolation_move_order_prioritises_under_separated_cargo():
    c=_isolation_controller()
    adjacent=[AdjacentContainer("GEN",10),AdjacentContainer("FAR",80),
              AdjacentContainer("ACID",20,frozenset({"8"})),
              AdjacentContainer("UNK",15,frozenset({"2.1"}))]
    inc=c.open_incident("I1","C1",_declared(),None,adjacent,[],D,"ops")
    order=inc.plan.move_order
    assert [a.container_id for a in order]==["UNK","ACID","GEN"]
    assert order[0].reason=="incompatible or under-separated cargo"
    assert order[-1].reason=="within isolation radius"

def test_isolation_replan_keeps_committed_actions():
    c=_isolation_controller()
    adjacent=[AdjacentContainer("A",10),AdjacentContainer("B",20)]
    inc=c.open_incident("I1","C1",_declared(),None,adjacent,[],D,"ops")
    c.commit_action("I1","B",D+timedelta(minutes=5),"ops")
    inc=c.replan("I1",_declared(),None,[AdjacentContainer("D",5)],[],D+timedelta(minutes=6),"ops")
    order=inc.plan.move_order
    assert order[0].container_id=="B" and order[0].state==ActionState.COMMITTED and order[0].sequence==2
    assert [a.container_id for a in order if a.state==ActionState.PLANNED]==["D"]
    with pytest.raises(StateTransitionError):
        c.commit_action("I1","B",D+timedelta(minutes=7),"ops")

def test_isolation_escalation_and_stand_down_are_audited():
    c=_isolation_controller()
    inc=c.open_incident("I1","C1",_declared(),None,[AdjacentContainer("A",10)],[],D,"ops")
    c.commit_action("I1","A",D+timedelta(minutes=1),"ops")
    with pytest.raises(StateTransitionError):
        c.stand_down("I1",D+timedelta(minutes=2),"ops","all clear")
    c.complete_action("I1","A",D+timedelta(minutes=3),"ops")
    c.escalate("I1",D+timedelta(minutes=4),"chief","fire spreading")
    assert inc.level==IncidentLevel.ESCALATED
    c.stand_down("I1",D+timedelta(minutes=5),"chief","area safe")
    assert inc.level==IncidentLevel.STOOD_DOWN
    trail=c.audit_trail("I1")
    assert [e.event for e in trail]==["opened","action_committed","action_completed","escalated","stood_down"]
    assert trail[-2].actor=="chief" and "fire spreading" in trail[-2].detail
    assert [e.revision for e in trail]==[1,2,3,4,5]
    with pytest.raises(StateTransitionError):
        c.replan("I1",_declared(),None,[],[],D+timedelta(minutes=6),"ops")

def test_isolation_fails_closed_without_guide():
    c=_isolation_controller()
    with pytest.raises(ValidationError):
        c.open_incident("I1","C1",_declared(classes=("6.1",)),None,[],[],D,"ops")

def test_maintenance_creates_due_order():
    mgr=MaintenanceManager();mgr.register_plan(MaintenancePlan("P",100));mgr.register_meter(EquipmentMeter("E",105,D))
    orders=mgr.open_due_orders(D+timedelta(hours=1))
    assert len(orders)==1 and orders[0].due_at_meter==100

def test_rail_assignment_respects_destination():
    planner=RailBlockPlanner([Wagon("W1",50000,40,"A"),Wagon("W2",50000,40,"B")])
    result=planner.assign([RailContainer("C1","B",20000,20)])
    assert result["C1"]=="W2"

def test_demurrage_skips_holiday():
    cal=BusinessCalendar({date(2026,1,2)},weekends_free=False)
    tariff=DemurrageTariff(0,(TariffBand(1,None,Money(Decimal("10"),"USD")),))
    calc=DemurrageCalculator(cal,tariff)
    total=calc.total(datetime(2026,1,1,tzinfo=UTC),datetime(2026,1,3,tzinfo=UTC))
    assert total.amount==Decimal("20.00")

def test_connection_slack():
    g=ConnectionGraph();g.add_call(ServiceCall("IN","S1",D,D+timedelta(hours=2)));g.add_call(ServiceCall("OUT","S2",D+timedelta(hours=5),D+timedelta(hours=8)))
    c=Connection("C","IN","OUT",timedelta(hours=4))
    g.add_connection(c)
    assert g.slack(c)==timedelta(hours=4)
